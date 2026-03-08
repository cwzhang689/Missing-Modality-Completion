# scripts/build_meta_from_iemocap.py
# 作用：从 IEMOCAP_full_release 目录解析出 4 类元数据，写到 .\Archive\meta.csv
# 用法（Windows 示例见文末）：
#   python scripts\build_meta_from_iemocap.py ^
#     --iemocap_root "D:\Datasets\IEMOCAP_full_release" ^
#     --out_csv .\Archive\meta.csv ^
#     --frames_root .\Archive\frames ^
#     --create_frames_dir

"""
python scripts\build_meta_from_iemocap.py ^
  --iemocap_root "D:\Datasets\IEMOCAP_full_release" ^
  --out_csv .\Archive\meta.csv ^
  --frames_root .\Archive\frames ^
  --create_frames_dir
"""

import argparse, os, re, csv, sys
from pathlib import Path

# 4-class (ang/hap(+exc)/sad/neu)
EMO_MAP_4 = {"ang": 0, "hap": 1, "exc": 1, "sad": 2, "neu": 3}

def parse_emo_file(fp: Path):
    """
    解析 dialog/EmoEvaluation/*.txt，返回 {utt_id: emo(str)}。
    兼容 Windows 换行、容错弱格式。
    """
    out = {}
    if not fp.exists(): return out
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\r\n") for ln in f]
    for i, ln in enumerate(lines):
        # 例：Ses01F_impro01_F000 [0.000 - 2.345]
        m = re.match(r'^([A-Za-z0-9_]+)\s*\[', ln)
        if not m:
            continue
        utt_id = m.group(1)
        emo = None
        # 情感一般在下一行；做 1~3 行的容错
        for j in range(1, 4):
            if i + j >= len(lines): break
            s = lines[i + j].strip().lower()
            # 常见是 ang/hap/exc/sad/neu
            if re.fullmatch(r'[a-z]{2,4}', s):
                emo = s
                break
        if emo:
            out[utt_id] = emo
    return out

def parse_transcript(fp: Path):
    """解析 dialog/transcriptions/*.txt -> {utt_id: text}"""
    out = {}
    if not fp.exists(): return out
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            # Ses01F_impro01_F000 [0.000 - 2.345]: content ...
            m = re.match(r'^([A-Za-z0-9_]+)\s+\[.*?\]\s*:\s*(.*)$', ln)
            if m:
                out[m.group(1)] = m.group(2).strip()
    return out

def guess_wav_path(sess_dir: Path, utt_id: str) -> Path:
    """
    IEMOCAP 分割音频常见位置：
    SessionX/sentences/wav/<conv>/<utt_id>.wav
    也有少量在 sentences/wav_impro/<conv>/ 或者 conv 命名变体。
    """
    conv = utt_id.rsplit("_", 1)[0]  # e.g., Ses01F_impro01
    candidates = [
        sess_dir / "sentences" / "wav" / conv / f"{utt_id}.wav",
        sess_dir / "sentences" / "wav_impro" / conv / f"{utt_id}.wav",
    ]

    # 再兜底一层：部分 corpora 会把 impro 目录名附加在上层
    wav_base = sess_dir / "sentences" / "wav"
    impro_base = sess_dir / "sentences" / "wav_impro"
    for base in (wav_base, impro_base):
        if base.exists():
            # 有时 conv 目录名不是严格等于前缀，枚举一层
            for d in base.glob(f"{conv}*"):
                cand = d / f"{utt_id}.wav"
                if cand.exists():
                    candidates.append(cand)

    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # 返回第一个候选（即便不存在，后面会统计缺失）

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iemocap_root", required=True,
                    help="IEMOCAP_full_release 根目录（包含 Session1..Session5）")
    # —— 按你的目录默认到 Archive 下 ——
    ap.add_argument("--out_csv", default=str(Path("./Archive/meta.csv")))
    ap.add_argument("--frames_root", default=str(Path("./Archive/frames")),
                    help="为每个 utt 预分配的人脸帧目录根（不存在可创建）")
    ap.add_argument("--create_frames_dir", action="store_true",
                    help="若指定，则为每条样本创建空的 frames 子目录")
    ap.add_argument("--keep_others", action="store_true",
                    help="保留非 {ang,hap/exc,sad,neu} 的样本（会被标注为 label=-1）")
    args = ap.parse_args()

    root = Path(args.iemocap_root)
    if not root.exists():
        print(f"[ERR] Not found: {root}", file=sys.stderr)
        sys.exit(1)

    rows = []
    dropped_emo = {}
    missing_wavs, missing_txt = [], []

    for s in range(1, 6):
        # 兼容大小写
        sess_dir = None
        for cand in [root / f"Session{s}", root / f"session{s}"]:
            if cand.exists():
                sess_dir = cand
                break
        assert sess_dir is not None, f"Missing Session{s} under {root}"

        eval_dir = sess_dir / "dialog" / "EmoEvaluation"
        tr_dir   = sess_dir / "dialog" / "transcriptions"
        assert eval_dir.exists(), f"Missing: {eval_dir}"
        assert tr_dir.exists(),   f"Missing: {tr_dir}"

        emo_all, txt_all = {}, {}
        for emo_fp in sorted(eval_dir.glob("*.txt")):
            emo_all.update(parse_emo_file(emo_fp))
        for tr_fp in sorted(tr_dir.glob("*.txt")):
            txt_all.update(parse_transcript(tr_fp))

        for utt_id, emo in emo_all.items():
            if emo in EMO_MAP_4:
                label = EMO_MAP_4[emo]
            else:
                if not args.keep_others:
                    dropped_emo[emo] = dropped_emo.get(emo, 0) + 1
                    continue
                label = -1

            text = txt_all.get(utt_id, "")
            if not text:
                missing_txt.append(utt_id)

            wav_path = guess_wav_path(sess_dir, utt_id)
            if not wav_path.exists():
                missing_wavs.append(str(wav_path))

            frames_dir = Path(args.frames_root) / utt_id
            if args.create_frames_dir:
                frames_dir.mkdir(parents=True, exist_ok=True)

            rows.append({
                "utt_id": utt_id,
                "text": text,
                "label": label,
                "session": s,
                "wav_path": str(wav_path.resolve()) if wav_path.is_absolute() else str(wav_path),
                "frames_dir": str(frames_dir.resolve()) if frames_dir.is_absolute() else str(frames_dir),
            })

    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id","text","label","session","wav_path","frames_dir"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    kept = sum(1 for r in rows if r["label"] != -1)
    others = sum(1 for r in rows if r["label"] == -1)
    print(f"[OK] meta.csv -> {outp} | kept(4-class)={kept} | others(non-4)={others}")

    if dropped_emo:
        print("[INFO] dropped emotions:", dropped_emo)
    if missing_wavs:
        miss = outp.parent / "missing_wavs.txt"
        miss.write_text("\n".join(missing_wavs), encoding="utf-8")
        print(f"[WARN] Missing wavs: {len(missing_wavs)} (list saved to {miss})")
    if missing_txt:
        mt = outp.parent / "missing_transcripts.txt"
        mt.write_text("\n".join(missing_txt), encoding="utf-8")
        print(f"[WARN] Missing transcripts: {len(missing_txt)} (list saved to {mt})")

if __name__ == "__main__":
    main()
