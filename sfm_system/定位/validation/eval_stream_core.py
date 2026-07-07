#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""串流定位驗證: 逐幀餵入驗證資料, XFeat + LighterGlue + PnP.

The default comparison uses the deployed base bundle against an updated bundle.
For a fair MegaLoc deployment comparison, pass --base-megaloc-cache so the old
bundle keeps its old geometry/keyframes but uses the same MegaLoc VPR type as
the updated bundle.
"""
import sys, os, time, argparse, json
import numpy as np, torch, cv2
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
from reloc_localizer_xfeat import load_xfeat, extract_xfeat, MegaLocQuery, bundle_vpr_kind
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
def loadb(p, megaloc_cache=None):
    b=torch.load(p,map_location='cpu',weights_only=False); rg=np.asarray(b['ref_global']).astype(np.float32)
    meta=dict(b.get('meta',{}))
    if megaloc_cache:
        dd=np.load(megaloc_cache,allow_pickle=True)
        desc=np.asarray(dd['desc'],np.float32)
        names=[str(x) for x in dd['names']]
        lut={n:i for i,n in enumerate(names)}
        missing=[n for n in b['ref_names'] if n not in lut]
        if missing:
            raise SystemExit(f"MegaLoc cache missing {len(missing)} ref names, first={missing[:3]}")
        rg=np.stack([desc[lut[n]] for n in b['ref_names']]).astype(np.float32)
        meta.update({"bundle_vpr":"megaloc","vpr":"megaloc-8448","vpr_input":322,
                     "global_descriptor_source":"MegaLoc eval override"})
    rg/=(np.linalg.norm(rg,axis=1,keepdims=True)+1e-9)
    return b['ref_names'],b['refs'],rg,bundle_vpr_kind(meta)
def stream(src,stride,resize_wh):
    """yield RGB frames in temporal order (true streaming)."""
    if str(src).lower().endswith(".mp4"):
        cap=cv2.VideoCapture(str(src)); i=0
        while True:
            ok,fr=cap.read()
            if not ok: break
            if i%stride==0:
                if resize_wh:
                    fr=cv2.resize(fr,resize_wh,interpolation=cv2.INTER_AREA)
                yield cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)
            i+=1
        cap.release()
    else:
        for k,p in enumerate(sorted(Path(src).glob("*.jpg"))):
            if k%stride==0:
                fr=cv2.imread(str(p),cv2.IMREAD_COLOR)
                if resize_wh:
                    fr=cv2.resize(fr,resize_wh,interpolation=cv2.INTER_AREA)
                yield cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)
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
def run(bundle,src,stride,resize_wh):
    inl=[]; retnew=[]; fail=0; maxfail=0
    for rgb in stream(src,stride,resize_wh):
        n,isnew=loc_one(bundle,rgb); inl.append(n); retnew.append(1 if isnew else 0)
        if n<ADD: fail+=1; maxfail=max(maxfail,fail)
        else: fail=0
    return np.array(inl),(np.mean(retnew) if retnew else 0),maxfail
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",default=BASE)
    ap.add_argument("--final",default=FINAL)
    ap.add_argument("--base-megaloc-cache")
    ap.add_argument("--test-dir",default=TEST_DIR)
    ap.add_argument("--stride",type=int,default=10)
    ap.add_argument("--resize",default="1280x720")
    ap.add_argument("--topk",type=int,default=30)
    ap.add_argument("--min-conf",type=float,default=0.1)
    ap.add_argument("--min-inliers",type=int,default=50)
    ap.add_argument("--out-json")
    args=ap.parse_args()
    global TOPK, MIN_CONF, ADD
    TOPK=int(args.topk); MIN_CONF=float(args.min_conf); ADD=int(args.min_inliers)
    resize_wh=None
    if args.resize:
        w,h=map(int,args.resize.lower().split("x"))
        resize_wh=(w,h)
    sets=[(p.stem, str(p), args.stride) for p in sorted(Path(args.test_dir).glob("*.MP4"))]
    if not sets:
        raise SystemExit(f"no MP4 files found under {args.test_dir}")
    bb=loadb(args.base,args.base_megaloc_cache); log(f"BASE {len(bb[0])} refs kind={bb[3]}")
    fb=loadb(args.final); log(f"FINAL {len(fb[0])} refs kind={fb[3]}")
    rows=[]
    print(f"\n{'set':10} | {'n':>4} | BASE succ / med  | FINAL succ / med | gain  | fail->ok reg | maxfail(B/F) | ret-new",flush=True)
    for name,src,stride in sets:
        rb,_,mfb=run(bb,src,stride,resize_wh); ru,rnw,mff=run(fb,src,stride,resize_wh)
        sb=float((rb>=ADD).mean()) if len(rb) else 0.0
        su=float((ru>=ADD).mean()) if len(ru) else 0.0
        rec=int(((rb<ADD)&(ru>=ADD)).sum()); reg=int(((rb>=ADD)&(ru<ADD)).sum())
        row={"set":name,"n":int(len(rb)),"base_success":sb,"final_success":su,
             "gain_pp":100*(su-sb),"base_median_inliers":float(np.median(rb)) if len(rb) else 0,
             "final_median_inliers":float(np.median(ru)) if len(ru) else 0,
             "fail_to_ok":rec,"ok_to_fail":reg,"base_max_fail_run":int(mfb),
             "final_max_fail_run":int(mff),"retrieved_new_fraction":float(rnw)}
        rows.append(row)
        log(f"{name:10} | {len(rb):4d} | {sb:5.1%} / {np.median(rb):4.0f} | {su:5.1%} / {np.median(ru):4.0f} | {100*(su-sb):+5.1f}pp | {rec:3d} / {reg:2d}    | {mfb:3d}/{mff:3d}     | {rnw:.0%}")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps({
            "resize":args.resize,
            "stride":args.stride,
            "topk":TOPK,
            "min_conf":MIN_CONF,
            "min_inliers":ADD,
            "rows":rows
        },indent=2),encoding="utf-8")
    if _WARN_SEEN: log(f"error-frame counters (not genuine 0-inlier): {_WARN_SEEN}")
    log("DONE")
if __name__=="__main__":
    main()
