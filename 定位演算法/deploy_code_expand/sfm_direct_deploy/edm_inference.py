"""Bounded GPU input/backbone reuse for the deployed, FP32 EDM matcher."""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np


@dataclass
class CachedImage:
    source: object
    pixels: object
    mask: object
    scale: object
    features: tuple | None = None


class CachedEDM:
    """Keep reference tensors on-device; reuse only image-independent features.

    A cache miss uses the original two-image backbone batch. This preserves its
    numerical shape instead of switching cold images to single-image kernels.
    The cross-image neck and both matching heads always run on the current pair.
    """

    def __init__(self, runtime, capacity: int):
        self.runtime = runtime
        self.capacity = capacity
        self.references: OrderedDict[int, CachedImage] = OrderedDict()
        self.query: CachedImage | None = None
        self.backbone_hits = 0

    def _upload(self, image) -> CachedImage:
        torch = self.runtime.torch
        device = self.runtime.device
        return CachedImage(
            image,
            torch.from_numpy(np.ascontiguousarray(image.pixels))[None, None]
            .to(device, dtype=torch.float32)
            .div_(255.0),
            torch.from_numpy(image.coarse_mask)[None].to(device),
            torch.tensor([image.scale], dtype=torch.float32, device=device),
        )

    def match(self, query, reference):
        runtime = self.runtime
        torch = runtime.torch
        if self.query is None or self.query.source is not query:
            self.query = self._upload(query)
        key = id(reference)
        ref = self.references.get(key)
        if ref is None:
            ref = self._upload(reference)
            self.references[key] = ref
            while len(self.references) > self.capacity:
                self.references.popitem(last=False)
        self.references.move_to_end(key)
        q = self.query
        batch = {
            "image0": q.pixels,
            "image1": ref.pixels,
            "mask0": q.mask,
            "mask1": ref.mask,
            "scale0": q.scale,
            "scale1": ref.scale,
        }
        matcher = runtime.matcher
        hook = None
        with torch.inference_mode():
            if q.features is not None and ref.features is not None:
                matcher._backbone_plan = True
                matcher._cache_fully_hit = True
                matcher._cached_pyramid = tuple(
                    torch.cat([left, right], dim=0)
                    for left, right in zip(q.features, ref.features, strict=True)
                )
                self.backbone_hits += 1
            else:

                def remember(_module, _inputs, outputs):
                    q.features = tuple(value[:1].clone() for value in outputs)
                    ref.features = tuple(value[1:].clone() for value in outputs)

                hook = matcher.backbone.register_forward_hook(remember)
            try:
                matcher(batch)
            finally:
                if hook is not None:
                    hook.remove()
                matcher._backbone_plan = None
                matcher._cache_fully_hit = False
                matcher._cached_pyramid = None
        return {
            name: batch[name].detach().cpu().numpy() for name in ("mkpts0_f", "mkpts1_f", "mconf")
        }
