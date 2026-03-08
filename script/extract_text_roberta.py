# scripts/extract_text_roberta.py
import argparse, pandas as pd, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

def mean_last_hidden_state(model, tok, text, device):
    t = tok(text, return_tensors="pt", truncation=True, max_length=256)
    out = model(**{k:v.to(device) for k,v in t.items()})
    h = out.last_hidden_state[0]               # [L, 768]
    return h.mean(0).detach().cpu().numpy()    # [768]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", required=True, help="csv: utt_id,text")
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--model", default="roberta-base")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModel.from_pretrained(args.model).to(device).eval()

    df = pd.read_csv(args.meta_csv)
    feats={}
    with torch.no_grad():
        for _,r in df.iterrows():
            uid = str(r["utt_id"])
            txt = str(r["text"])
            v = mean_last_hidden_state(mdl, tok, txt, device)  # [768]
            feats[uid] = v.astype(np.float32)
    np.savez_compressed(args.out_npz, **feats)
    print(f"[OK] saved text npz -> {args.out_npz}  (N={len(feats)})")

if __name__=="__main__":
    main()
