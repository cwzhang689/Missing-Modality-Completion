# scripts/extract_visual_libreface.py
"""
meta_csv: utt_id,frames_dir
frames_dir 下放该话语的对齐人脸帧（jpg/png）。若未对齐，脚本会用 MTCNN 做简单检测裁剪。
输出：每个 utt 取均匀 20 帧，每帧过 ArcFace(InsightFace) 得到 512 维，最后 [20,512]
"""
import argparse, pandas as pd, numpy as np, cv2, torch
from pathlib import Path
from facenet_pytorch import MTCNN
import insightface
from insightface.app import FaceAnalysis

def load_detector(device):
    mtcnn = MTCNN(image_size=224, margin=20, post_process=True, device=device)
    return mtcnn

def load_arcface():
    app = FaceAnalysis(name="buffalo_l")  # r50/r100
    app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(256,256))
    return app

def read_image(fp:str):
    img = cv2.imread(fp)
    if img is None: return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def crop_face(mtcnn, img):
    # 返回对齐人脸 112x112（InsightFace 仍可用）
    try:
        x = mtcnn(img, return_prob=False)
        if x is None: return None
        return (x.permute(1,2,0).numpy()*255).astype(np.uint8)
    except:
        return None

def uniform_pick(paths, K=20):
    if len(paths)<=0: return []
    idx = np.linspace(0, len(paths)-1, K).round().astype(int).tolist()
    return [paths[i] for i in idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", required=True, help="csv: utt_id,frames_dir")
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--target_T", type=int, default=20)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = load_detector(device)
    arc = load_arcface()

    df = pd.read_csv(args.meta_csv)
    feats={}
    for _,r in df.iterrows():
        uid = str(r["utt_id"])
        d = Path(str(r["frames_dir"]))
        frames = sorted([str(p) for p in d.glob("*.jpg")] + [str(p) for p in d.glob("*.png")])
        sel = uniform_pick(frames, args.target_T)
        xs=[]
        for fp in sel:
            img = read_image(fp)
            if img is None: continue
            # 若已经是对齐人脸，可直接喂 arc.get
            face = crop_face(mtcnn, img)
            if face is None: face = img
            faces = arc.get(face)  # 返回检测结果；若无则对整图
            if len(faces)==0:
                faces = arc.get(img)
            if len(faces)==0:
                continue
            emb = faces[0]["embedding"]  # [512]
            xs.append(emb.astype(np.float32))
        if len(xs)==0:
            xs = [np.zeros(512, dtype=np.float32)]*args.target_T
        if len(xs)<args.target_T:
            xs = xs + [xs[-1]]*(args.target_T-len(xs))
        feats[uid] = np.stack(xs[:args.target_T],0)
    np.savez_compressed(args.out_npz, **feats)
    print(f"[OK] saved visual npz -> {args.out_npz}  (N={len(feats)})")

if __name__=="__main__":
    main()
