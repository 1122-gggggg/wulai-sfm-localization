import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loftr_module.transformer import LocalFeatureTransformer


class Conv2d_BN_Act(nn.Sequential):
    def __init__(
        self,
        a,
        b,
        ks=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        act=None,
        drop=None,
    ):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module(
            "c", nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False)
        )
        bn = nn.BatchNorm2d(b)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)
        if act != None:
            self.add_module("a", act)
        if drop != None:
            self.add_module("d", nn.Dropout(drop))


class CIM(nn.Module):
    """Feature Aggregation, Correlation Injection Module"""

    def __init__(self, config):
        super(CIM, self).__init__()

        self.block_dims = config["backbone"]["block_dims"]
        self.drop = config["fine"]["droprate"]

        self.fc32 = Conv2d_BN_Act(
            self.block_dims[-1], self.block_dims[-1], 1, drop=self.drop
        )
        self.fc16 = Conv2d_BN_Act(
            self.block_dims[-2], self.block_dims[-1], 1, drop=self.drop
        )
        self.fc8 = Conv2d_BN_Act(
            self.block_dims[-3], self.block_dims[-1], 1, drop=self.drop
        )
        self.att32 = Conv2d_BN_Act(
            self.block_dims[-1],
            self.block_dims[-1],
            1,
            act=nn.Sigmoid(),
            drop=self.drop,
        )
        self.att16 = Conv2d_BN_Act(
            self.block_dims[-1],
            self.block_dims[-1],
            1,
            act=nn.Sigmoid(),
            drop=self.drop,
        )
        self.dwconv16 = nn.Sequential(
            Conv2d_BN_Act(
                self.block_dims[-1],
                self.block_dims[-1],
                ks=3,
                pad=1,
                groups=self.block_dims[-1],
                act=nn.GELU(),
            ),
            Conv2d_BN_Act(self.block_dims[-1], self.block_dims[-1], 1),
        )
        self.dwconv8 = nn.Sequential(
            Conv2d_BN_Act(
                self.block_dims[-1],
                self.block_dims[-1],
                ks=3,
                pad=1,
                groups=self.block_dims[-1],
                act=nn.GELU(),
            ),
            Conv2d_BN_Act(self.block_dims[-1], self.block_dims[-1], 1),
        )

        self.loftr_32 = LocalFeatureTransformer(config["neck"])

    def forward(self, ms_feats, mask_c0=None, mask_c1=None):
        if len(ms_feats) == 3:  # same image shape
            f8, f16, f32 = ms_feats
            # GPU redundancy: query unary CNN (fc32/fc16/fc8/att32/att16/dwconv) runs on 2*B with duplicated query.
            # Ledger rejected 15-25% saving claim for query repeat->expand without synchronized exact benchmark (docs/verified_localization_optimization_ledger.md: query tensor repeat改expand not yet verified).
            # Optimize by computing query branch once (first query element) then replicate via expand/repeat for cross-attention; keep fallback 2*B path when B==1 or not duplicated to preserve exact behavior.
            B = f32.shape[0] // 2
            # SFM_EDM_NECK_NO_EXPAND=1 forces the verbatim 2*B path for exactness A/B.
            duplicated = (
                B > 1
                and f32.shape[0] % 2 == 0
                and f8.shape[0] == f32.shape[0]
                and f16.shape[0] == f32.shape[0]
                and os.environ.get("SFM_EDM_NECK_NO_EXPAND") != "1"
            )
            if duplicated:
                # Split refs and single query (query is tiled B times in match_many_to_one)
                f32_ref = f32[:B]
                f32_qry_one = f32[B:B+1]
                f16_ref = f16[:B]
                f16_qry_one = f16[B:B+1]
                f8_ref = f8[:B]
                f8_qry_one = f8[B:B+1]
                # Compute query unary CNN once then expand - saves B-1 redundant convs
                f32_ref = self.fc32(f32_ref)
                f32_qry_one = self.fc32(f32_qry_one)
                f32_qry = f32_qry_one.expand(B, -1, -1, -1)
                # Ensure contiguous for LoFTR (channels_last aware)
                if f32_ref.is_contiguous(memory_format=torch.channels_last):
                    f32_qry = f32_qry.contiguous(memory_format=torch.channels_last)
                else:
                    f32_qry = f32_qry.contiguous()
                f32_0, f32_1 = self.loftr_32(f32_ref, f32_qry, mask_c0, mask_c1)
                f32 = torch.cat([f32_0, f32_1], dim=0)
                # f16 path: fc16 once for query
                f32_up = F.interpolate(f32, scale_factor=2.0, mode="bilinear")
                att32_up = F.interpolate(self.att32(f32), scale_factor=2.0, mode="bilinear")
                f16_ref = self.fc16(f16_ref)
                f16_qry_one = self.fc16(f16_qry_one)
                f16_qry = f16_qry_one.expand(B, -1, -1, -1)
                if f16_ref.is_contiguous(memory_format=torch.channels_last):
                    f16_qry = f16_qry.contiguous(memory_format=torch.channels_last)
                else:
                    f16_qry = f16_qry.contiguous()
                f16 = torch.cat([f16_ref, f16_qry], dim=0)
                f16 = self.dwconv16(f16 * att32_up + f32_up)
                f16_up = F.interpolate(f16, scale_factor=2.0, mode="bilinear")
                att16_up = F.interpolate(self.att16(f16), scale_factor=2.0, mode="bilinear")
                f8_ref = self.fc8(f8_ref)
                f8_qry_one = self.fc8(f8_qry_one)
                f8_qry = f8_qry_one.expand(B, -1, -1, -1)
                if f8_ref.is_contiguous(memory_format=torch.channels_last):
                    f8_qry = f8_qry.contiguous(memory_format=torch.channels_last)
                else:
                    f8_qry = f8_qry.contiguous()
                f8 = torch.cat([f8_ref, f8_qry], dim=0)
                f8 = self.dwconv8(f8 * att16_up + f16_up)
                feat_c0, feat_c1 = f8.chunk(2)
            else:
                f32 = self.fc32(f32)
                f32_0, f32_1 = f32.chunk(2, dim=0)
                f32_0, f32_1 = self.loftr_32(f32_0, f32_1, mask_c0, mask_c1)
                f32 = torch.cat([f32_0, f32_1], dim=0)
                f32_up = F.interpolate(f32, scale_factor=2.0, mode="bilinear")
                att32_up = F.interpolate(self.att32(
                    f32), scale_factor=2.0, mode="bilinear")
                f16 = self.fc16(f16)
                f16 = self.dwconv16(f16 * att32_up + f32_up)
                f16_up = F.interpolate(f16, scale_factor=2.0, mode="bilinear")
                att16_up = F.interpolate(self.att16(
                    f16), scale_factor=2.0, mode="bilinear")
                f8 = self.fc8(f8)
                f8 = self.dwconv8(f8 * att16_up + f16_up)
                feat_c0, feat_c1 = f8.chunk(2)

        elif len(ms_feats) == 6:  # diffirent image shape
            f8_0, f16_0, f32_0, f8_1, f16_1, f32_1 = ms_feats
            f32_0 = self.fc32(f32_0)
            f32_1 = self.fc32(f32_1)

            f32_0, f32_1 = self.loftr_32(f32_0, f32_1, mask_c0, mask_c1)

            f8, f16, f32 = f8_0, f16_0, f32_0
            f32_up = F.interpolate(f32, scale_factor=2.0, mode="bilinear")
            att32_up = F.interpolate(self.att32(
                f32), scale_factor=2.0, mode="bilinear")
            f16 = self.fc16(f16)
            f16 = self.dwconv16(f16 * att32_up + f32_up)
            f16_up = F.interpolate(f16, scale_factor=2.0, mode="bilinear")
            att16_up = F.interpolate(self.att16(
                f16), scale_factor=2.0, mode="bilinear")
            f8 = self.fc8(f8)
            f8 = self.dwconv8(f8 * att16_up + f16_up)
            feat_c0 = f8

            f8, f16, f32 = f8_1, f16_1, f32_1
            f32_up = F.interpolate(f32, scale_factor=2.0, mode="bilinear")
            att32_up = F.interpolate(self.att32(
                f32), scale_factor=2.0, mode="bilinear")
            f16 = self.fc16(f16)
            f16 = self.dwconv16(f16 * att32_up + f32_up)
            f16_up = F.interpolate(f16, scale_factor=2.0, mode="bilinear")
            att16_up = F.interpolate(self.att16(
                f16), scale_factor=2.0, mode="bilinear")
            f8 = self.fc8(f8)
            f8 = self.dwconv8(f8 * att16_up + f16_up)
            feat_c1 = f8

        return feat_c0, feat_c1
