import json, argparse, os

CUT = ["Human:", "Assistant:", "User:"]

def cut_text(t: str) -> str:
    if not t:
        return ""
    s = t
    earliest = None
    for m in CUT:
        p = s.find(m)
        if p != -1:
            earliest = p if earliest is None else min(earliest, p)
    if earliest is not None:
        s = s[:earliest]
    return s.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default="data/dpo_pairs.clean.jsonl")
    ap.add_argument("--out_path", default="data/dpo_pairs.clean2.jsonl")
    ap.add_argument("--min_chars", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    kept = 0
    dropped = 0
    trimmed = 0

    with open(args.in_path, "r", encoding="utf-8") as fin, open(args.out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            r0 = (rec.get("rejected") or "").strip()
            r1 = cut_text(r0)
            if r1 != r0:
                trimmed += 1
            if len(r1) < args.min_chars:
                dropped += 1
                continue
            rec["rejected"] = r1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Done. kept={kept}, dropped={dropped}, trimmed={trimmed}")
    print(f"Saved to: {args.out_path}")

if __name__ == "__main__":
    main()
