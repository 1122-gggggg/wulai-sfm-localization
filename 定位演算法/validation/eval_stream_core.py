#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""串流定位驗證: 逐幀餵入驗證資料, XFeat + LighterGlue + PnP.

The default comparison uses the deployed base bundle against an updated bundle.
For a fair MegaLoc deployment comparison, pass --base-megaloc-cache so the old
bundle keeps its old geometry/keyframes but uses the same MegaLoc VPR type as
the updated bundle.
"""
import sys, os, time, argparse, json
import numpy as np
from pathlib import Path


def find_system_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if p.name == "sfm_system":
            return p
    env = os.environ.get("SFM_SYSTEM_ROOT")
    if env:
        return Path(env)
    raise SystemExit(f"could not locate sfm_system root from {start}; set SFM_SYSTEM_ROOT")


SYSTEM_ROOT = find_system_root(Path(__file__).resolve())
LOC_ROOT = SYSTEM_ROOT / "定位"
DEPLOY_DIR = LOC_ROOT / "deploy_code" / "sfm_glomap_deploy"
SFM_GLOMAP = LOC_ROOT / "source" / "sfm_glomap"
if DEPLOY_DIR.exists():
    sys.path.insert(0, str(DEPLOY_DIR))
else:
    sys.path.insert(0, str(SFM_GLOMAP / "deploy"))
sys.path.insert(0, str(SFM_GLOMAP / "scripts"))
from reloc_localizer_xfeat import (
    MegaLocQuery,
    bundle_vpr_kind,
    extract_xfeat,
    load_verified_bundle,
    load_xfeat,
)
from megaloc_cache import load_megaloc_cache
from stream_integrity import StreamAudit, ffprobe_frame_count, iter_rgb_frames
import pycolmap
DEV="cuda"; QK=4096; ADD=50; TOPK=30; MIN_CONF=0.1
BASE=str(LOC_ROOT / "bundles" / "base_reloc_map_xfeat_tri.pt")
FINAL=str(LOC_ROOT / "bundles" / "current_reloc_map_updated_v3.pt")
TEST_DIR=str(SYSTEM_ROOT / "更新地圖" / "inputs" / "補拍影片" / "test")
FIXED_INTRINSICS={
    (2688,1512): [1955.5,1344.0,756.0,0.0020],
    (1920,1080): [1400.0,960.0,540.0,0.0015],
    (1280,720): [936.5,640.0,360.0,0.0035],
}
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
_FRAME=0                 # global frame counter for diagnostics
_WARN_SEEN={}            # tag -> count of error frames (distinct from genuine 0-inlier)
def _warn(tag,frame,e):
    _WARN_SEEN[tag]=_WARN_SEEN.get(tag,0)+1
    c=_WARN_SEEN[tag]
    if c<=5 or c%100==0:  # rate-limit: first few, then every 100th
        print(f"[eval_stream_core] {tag} frame={frame} n={c}: {e!r}",file=sys.stderr,flush=True)
xf=load_xfeat(QK)
_MEG=None
def fixed_intrinsics(W,H):
    if (W,H) in FIXED_INTRINSICS:
        return FIXED_INTRINSICS[(W,H)]
    f=1955.5*float(W)/2688.0
    return [f,float(W)/2.0,float(H)/2.0,0.002]
def loadb(p, megaloc_cache=None, expected_sha256=None, megaloc_meta=None):
    b=load_verified_bundle(p,expected_sha256); rg=np.asarray(b['ref_global']).astype(np.float32)
    meta=dict(b.get('meta',{}))
    if megaloc_cache:
        rg=load_megaloc_cache(megaloc_cache,list(b['ref_names']),megaloc_meta)
        meta.update({"bundle_vpr":"megaloc","vpr":"megaloc-8448","vpr_input":322,
                     "global_descriptor_source":"MegaLoc eval override"})
    rg/=(np.linalg.norm(rg,axis=1,keepdims=True)+1e-9)
    return b['ref_names'],b['refs'],rg,bundle_vpr_kind(meta)
def collect(q,qk,rn,rf,rg,qg,topk=None):
    topk = TOPK if topk is None else int(topk)
    idx=np.argsort(-(rg@qg))[:topk]; best={}
    for ti in idx:
        r=rf[rn[ti]]; rff={k:(v.to(DEV) if hasattr(v,'to') else v) for k,v in r['feats'].items()}
        try:_,_,mi=xf.match_lighterglue(q,rff,min_conf=MIN_CONF)     # XFeat + LightGlue
        except Exception as e:_warn("match_error",_FRAME,e);continue
        if mi is None or len(mi)==0:continue
        mi=np.asarray(mi.detach().cpu() if hasattr(mi,'detach') else mi); rx=np.asarray(r['xyz'])
        for qi,rj in mi:
            v=rx[int(rj)];qi=int(qi)
            if np.isfinite(v).all() and qi not in best: best[qi]=(qk[qi],v)
    if not best:return None,None
    return np.array([x[0] for x in best.values()]),np.array([x[1] for x in best.values()])
def global_desc(bundle,rgb):
    global _MEG
    if _MEG is None:
        _MEG=MegaLocQuery(DEV)
    return _MEG.extract_one(rgb)
def loc_one(bundle,rgb):
    global _FRAME; _FRAME+=1
    rn,rf,rg,kind=bundle; H,W=rgb.shape[:2]
    q=extract_xfeat(xf,rgb,QK); qk=np.asarray(q['keypoints'].detach().cpu())
    qg=global_desc(bundle,rgb)
    P2,P3=collect(q,qk,rn,rf,rg,qg); n=0
    if P2 is not None and len(P2)>=6:
        params=fixed_intrinsics(W,H)
        cam=pycolmap.Camera.create_from_model_id(0,pycolmap.CameraModelId.SIMPLE_RADIAL,params[0],W,H)
        cam.params=params
        try:
            r=pycolmap.estimate_and_refine_absolute_pose(P2.astype(float),P3.astype(float),cam)
            if r:n=r.get('num_inliers',0)
        except Exception as e:_warn("pose_error",_FRAME,e)
    top1=int(np.argmax(rg@qg)); isnew=rn[top1].startswith(("P1210121","P1220122","P1240124","P1250125"))
    return n,isnew
def run(bundle,src,stride,resize_wh,expected_raw_frames,expected_source,min_sampled_frames):
    inl=[]; retnew=[]; fail=0; maxfail=0
    audit=StreamAudit(expected_raw_frames=expected_raw_frames,
                      expected_source=expected_source,
                      integrity_min_sampled_frames=min_sampled_frames)
    for rgb in iter_rgb_frames(src,stride,resize_wh,audit):
        n,isnew=loc_one(bundle,rgb); inl.append(n); retnew.append(1 if isnew else 0)
        if n<ADD: fail+=1; maxfail=max(maxfail,fail)
        else: fail=0
    return np.array(inl),(np.mean(retnew) if retnew else 0),maxfail,audit


def parse_expected_frame_specs(specs):
    result={}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--expected-raw-frames must be FILE=COUNT, got {spec!r}")
        name,value=spec.rsplit("=",1)
        try: count=int(value)
        except ValueError: raise SystemExit(f"invalid expected frame count: {spec!r}")
        if not name or count<=0 or name in result:
            raise SystemExit(f"invalid or duplicate expected frame count: {spec!r}")
        result[name]=count
    return result


def resolve_expected_frames(path, expected_map):
    matches=[key for key in (str(path),Path(path).name,Path(path).stem) if key in expected_map]
    probed_count,probed_source=ffprobe_frame_count(path)
    if matches:
        supplied=expected_map[matches[0]]
        if probed_count is not None and supplied!=probed_count:
            raise SystemExit(
                f"trusted frame count conflict for {path}: cli={supplied} "
                f"{probed_source}={probed_count}"
            )
        return supplied,f"cli:{matches[0]}+{probed_source or 'no_ffprobe'}",matches[0]
    return probed_count,probed_source,None


def combined_stream_fields(base_audit,final_audit,paired,stride,min_sampled_frames):
    errors=base_audit.decode_errors+final_audit.decode_errors
    if (base_audit.decoded_raw_frames!=final_audit.decoded_raw_frames
            or base_audit.sampled_frames!=final_audit.sampled_frames):
        errors+=1
    expected=base_audit.expected_raw_frames
    if expected!=final_audit.expected_raw_frames:
        errors+=1; expected=None
    if base_audit.decode_complete is False or final_audit.decode_complete is False:
        complete=False
    elif base_audit.decode_complete is True and final_audit.decode_complete is True:
        complete=True
    else:
        complete=None
    reported=(base_audit.reported_raw_frames
              if base_audit.reported_raw_frames==final_audit.reported_raw_frames else None)
    return {
        "capture_opened":base_audit.capture_opened and final_audit.capture_opened,
        "reported_raw_frames":reported,
        "expected_raw_frames":expected,
        "expected_frame_source":base_audit.expected_source,
        "decoded_raw_frames":min(base_audit.decoded_raw_frames,final_audit.decoded_raw_frames),
        "sampled_frames":paired,
        "expected_sampled_frames":None if expected is None else (expected+stride-1)//stride,
        "decode_complete":complete,
        "decode_errors":errors,
        "integrity_min_sampled_frames":min_sampled_frames,
        "base_stream":base_audit.as_dict(),
        "final_stream":final_audit.as_dict(),
    }
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",default=BASE)
    ap.add_argument("--final",default=FINAL)
    ap.add_argument("--base-megaloc-cache")
    ap.add_argument("--base-megaloc-meta",
                    help="required binding JSON when --base-megaloc-cache is a legacy NPY")
    ap.add_argument("--base-sha256", default="",
                    help="trusted SHA-256 for a non-package base bundle")
    ap.add_argument("--final-sha256", default="",
                    help="trusted SHA-256 for a non-package final bundle")
    ap.add_argument("--test-dir",default=TEST_DIR)
    ap.add_argument("--stride",type=int,default=10)
    ap.add_argument("--resize",default="1280x720")
    ap.add_argument("--topk",type=int,default=30)
    ap.add_argument("--min-conf",type=float,default=0.1)
    ap.add_argument("--min-inliers",type=int,default=50)
    ap.add_argument("--expected-raw-frames",action="append",default=[],metavar="FILE=COUNT",
                    help="trusted intended raw-frame count; match by path, filename, or stem")
    ap.add_argument("--min-sampled-frames",type=int,default=30,
                    help="minimum sampled frames required even when decoding is complete")
    ap.add_argument("--out-json")
    args=ap.parse_args()
    global TOPK, MIN_CONF, ADD
    TOPK=int(args.topk); MIN_CONF=float(args.min_conf); ADD=int(args.min_inliers)
    resize_wh=None
    if args.resize:
        w,h=map(int,args.resize.lower().split("x"))
        resize_wh=(w,h)
    test_dir=Path(args.test_dir)
    paths=sorted(path for path in test_dir.iterdir()
                 if path.is_file() and path.suffix.lower()==".mp4") if test_dir.is_dir() else []
    if not paths:
        raise SystemExit(f"no MP4 files found under {args.test_dir}")
    if args.stride<=0 or args.min_sampled_frames<=0:
        raise SystemExit("--stride and --min-sampled-frames must be positive")
    expected_map=parse_expected_frame_specs(args.expected_raw_frames)
    used_expected=set();sets=[]
    for path in paths:
        expected,source,used=resolve_expected_frames(path,expected_map)
        if used: used_expected.add(used)
        if expected is None and args.min_sampled_frames<=0:
            raise SystemExit(
                f"cannot prove stream completeness for {path}; pass --expected-raw-frames FILE=COUNT "
                "or --min-sampled-frames"
            )
        sets.append((path.stem,str(path),args.stride,expected,source))
    unused=set(expected_map)-used_expected
    if unused:
        raise SystemExit(f"unused --expected-raw-frames keys: {sorted(unused)}")
    bb=loadb(args.base,args.base_megaloc_cache,args.base_sha256 or None,args.base_megaloc_meta); log(f"BASE {len(bb[0])} refs kind={bb[3]}")
    fb=loadb(args.final,expected_sha256=args.final_sha256 or None); log(f"FINAL {len(fb[0])} refs kind={fb[3]}")
    rows=[]
    print(f"\n{'set':10} | {'n':>4} | BASE succ / med  | FINAL succ / med | gain  | fail->ok reg | maxfail(B/F) | ret-new",flush=True)
    for name,src,stride,expected,expected_source in sets:
        rb,_,mfb,base_audit=run(bb,src,stride,resize_wh,expected,expected_source,args.min_sampled_frames)
        ru,rnw,mff,final_audit=run(fb,src,stride,resize_wh,expected,expected_source,args.min_sampled_frames)
        base_n,final_n=len(rb),len(ru)
        if len(rb) != len(ru):
            _warn("frame_count_mismatch", name, f"base={len(rb)} final={len(ru)}")
        paired=min(len(rb),len(ru)); rb=rb[:paired]; ru=ru[:paired]
        stream_fields=combined_stream_fields(
            base_audit,final_audit,paired,stride,args.min_sampled_frames)
        sb=float((rb>=ADD).mean()) if len(rb) else 0.0
        su=float((ru>=ADD).mean()) if len(ru) else 0.0
        rec=int(((rb<ADD)&(ru>=ADD)).sum()); reg=int(((rb>=ADD)&(ru<ADD)).sum())
        row={"set":name,"n":int(paired),"base_n":int(base_n),"final_n":int(final_n),
             "base_success":sb,"final_success":su,
             "gain_pp":100*(su-sb),"base_median_inliers":float(np.median(rb)) if len(rb) else 0,
             "final_median_inliers":float(np.median(ru)) if len(ru) else 0,
             "fail_to_ok":rec,"ok_to_fail":reg,"base_max_fail_run":int(mfb),
             "final_max_fail_run":int(mff),"retrieved_new_fraction":float(rnw),
             **stream_fields}
        rows.append(row)
        log(f"{name:10} | {len(rb):4d} | {sb:5.1%} / {np.median(rb):4.0f} | {su:5.1%} / {np.median(ru):4.0f} | {100*(su-sb):+5.1f}pp | {rec:3d} / {reg:2d}    | {mfb:3d}/{mff:3d}     | {rnw:.0%}")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps({
            "resize":args.resize,
            "stride":args.stride,
            "topk":TOPK,
            "min_conf":MIN_CONF,
            "min_inliers":ADD,
            "min_sampled_frames":args.min_sampled_frames,
            "error_frame_counters":dict(sorted(_WARN_SEEN.items())),
            "rows":rows
        },indent=2),encoding="utf-8")
    if _WARN_SEEN: log(f"error-frame counters (not genuine 0-inlier): {_WARN_SEEN}")
    log("DONE")
if __name__=="__main__":
    main()
