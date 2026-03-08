# scripts/build_iemocap_kfold_pkls.py
"""
输入：
  meta.csv:  utt_id,text,label,session,wav_path,frames_dir
  text.npz:  {utt_id: [768]}
  audio.npz: {utt_id: [T=20,768]}
  visual.npz:{utt_id: [T=20,512]}
输出：
  ./Archive/iemocap_S{1..5}.pkl  每份里自带 train/valid/test 三拆分（test=该 Session 全部）
label 取值 {0..3}，将转成 one-hot，便于你的 train.py 自动识别 multiclass。
"""
import argparse, pandas as pd, numpy as np, pickle, os
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path

def one_hot(y, C=4):
    m = np.zeros((len(y), C), dtype=np.float32)
    m[np.arange(len(y)), y] = 1.0
    return m

def pack_split(df_split, text_z, audio_z, visual_z):
    X_t, X_a, X_v, Y = [], [], [], []
    for _,r in df_split.iterrows():
        uid = str(r["utt_id"])
        t = text_z.get(uid, None)
        a = audio_z.get(uid, None)
        v = visual_z.get(uid, None)
        if t is None or a is None or v is None: continue
        X_t.append(t.astype(np.float32))            # [768]
        X_a.append(a.astype(np.float32))            # [20,768]
        X_v.append(v.astype(np.float32))            # [20,512]
        Y.append(int(r["label"]))
    X_t = np.stack(X_t, 0) if X_t else np.zeros((0,768), np.float32)
    X_a = np.stack(X_a, 0) if X_a else np.zeros((0,20,768), np.float32)
    X_v = np.stack(X_v, 0) if X_v else np.zeros((0,20,512), np.float32)
    Y   = one_hot(Y, C=4) if Y else np.zeros((0,4), np.float32)
    # 缺失模态标记（全 1，表示都可用；你的 diffusion 会自己用到 idx）
    missing = np.zeros((X_t.shape[0],), np.int64)
    return {"text":X_t, "audio":X_a, "visual":X_v, "label":Y, "missing":missing}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", required=True)
    ap.add_argument("--text_npz", required=True)
    ap.add_argument("--audio_npz", required=True)
    ap.add_argument("--visual_npz", required=True)
    ap.add_argument("--out_dir", default="./Archive")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="从训练会话里再 stratified 抽验证集")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.meta_csv)
    zt = dict(np.load(args.text_npz))
    za = dict(np.load(args.audio_npz))
    zv = dict(np.load(args.visual_npz))

    for S in [1,2,3,4,5]:
        df_test = df[df.session==S].copy()
        df_train_all = df[df.session!=S].copy()
        # stratified 验证集
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=2025)
        idx_train, idx_val = next(sss.split(df_train_all, df_train_all["label"]))
        df_train = df_train_all.iloc[idx_train]
        df_val   = df_train_all.iloc[idx_val]

        pack = {
            "train": pack_split(df_train, zt, za, zv),
            "valid": pack_split(df_val,   zt, za, zv),
            "test":  pack_split(df_test,  zt, za, zv),
            "meta": {
                "num_classes": 4,
                "classes": ["ang","hap","sad","neu"],
                "dims": {"text": 768, "audio": 768, "visual": 512},
                "seq_len": {"text": 1, "audio": 20, "visual": 20},
                "note": "text=RoBERTa-mean(768), audio=W2V2-meanpool->20x768, visual=ArcFace 20x512"
            }
        }
        outp = os.path.join(args.out_dir, f"iemocap_S{S}.pkl")
        with open(outp, "wb") as f:
            pickle.dump(pack, f)
        print(f"[OK] {outp} | train={len(df_train)} val={len(df_val)} test={len(df_test)}")

if __name__=="__main__":
    main()
