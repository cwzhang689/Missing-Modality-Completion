# scripts/extract_audio_w2v2.py
import argparse, pandas as pd, numpy as np, soundfile as sf, torch, librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from pathlib import Path

def uniform_time_pool(x, target_T=20):
    # x: [T, D] -> 均匀采样/均值池化到 target_T
    T, D = x.shape
    if T == target_T: return x
    idx = np.linspace(0, T-1, target_T)
    idx0 = np.floor(idx).astype(int)
    idx1 = np.clip(idx0+1, 0, T-1)
    w = (idx - idx0)[:, None]
    return (1-w)*x[idx0] + w*x[idx1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", required=True, help="csv: utt_id,wav_path")
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--model", default="facebook/wav2vec2-base-960h")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--target_T", type=int, default=20)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = Wav2Vec2Processor.from_pretrained(args.model)
    mdl  = Wav2Vec2Model.from_pretrained(args.model).to(device).eval()

    df = pd.read_csv(args.meta_csv)
    feats={}
    with torch.no_grad():
        for _,r in df.iterrows():
            uid = str(r["utt_id"])
            wav = str(r["wav_path"])
            x, sr = sf.read(wav)
            if sr != args.sr:
                x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=args.sr)
            if x.ndim==2:
                x = x.mean(-1)
            inputs = proc(x, sampling_rate=args.sr, return_tensors="pt", padding=True)
            z = mdl(inputs.input_values.to(device)).last_hidden_state.squeeze(0).cpu().numpy()  # [T, 768]
            z20 = uniform_time_pool(z, args.target_T)
            feats[uid] = z20.astype(np.float32)
    np.savez_compressed(args.out_npz, **feats)
    print(f"[OK] saved audio npz -> {args.out_npz}  (N={len(feats)})")

if __name__=="__main__":
    main()
