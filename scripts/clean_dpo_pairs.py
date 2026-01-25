import json, argparse, os, re

# Cut markers anywhere in the rejected text (not only after newline)
CUT_MARKERS = [
    r"###\s*Instruction:",
    r"\bHuman:\b",
    r"\bUser:\b",
    r"\bAssistant:\b",
    r"You are an AI assistant",
]

def cut_at_markers(text: str) -> str:
    if not text:
        return ""

    t = text.strip()

    # 1) If model starts a new instruction / chat roles anywhere -> cut from earliest marker
    earliest = None
    for pat in CUT_MARKERS:
        m = re.search(pat, t)
        if m:
            earliest = m.start() if earliest is None else min(earliest, m.start())
    if earliest is not None:
        t = t[:earliest].strip()

    # 2) If response header repeats, keep only first answer part
    # (some generations contain "### Response:" again)
    if "### Response:" in t:
        # keep only text before the first repeated response header occurrence AFTER some content
        parts = t.split("### Response:")
        if len(parts) >= 2:
            # keep everything before the first repeated header (i.e., parts[0])
            # but if parts[0] is empty, take the next non-empty part
            base = parts[0].strip()
            if not base:
                for p in parts[1:]:
                    p = p.strip()
                    if p:
                        base = p
                        break
            t = base.strip()

    # 3) Normalize whitespace
    t = re.sub(r"\s+\Z", "", t)
    return t.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default="data/dpo_pairs.jsonl")
    ap.add_argument("--out_path", default="data/dpo_pairs.clean.jsonl")
    ap.add_argument("--min_chars", type=int, default=20)
    ap.add_argument("--max_chars", type=int, default=2000)
    args = ap.parse_args()

    kept = 0
    dropped = 0
    trimmed = 0

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    with open(args.in_path, "r", encoding="utf-8") as fin, open(args.out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            rej0 = (rec.get("rejected") or "").strip()
            rej1 = cut_at_markers(rej0)

            if rej1 != rej0:
                trimmed += 1

            if len(rej1) < args.min_chars:
                dropped += 1
                continue

            if len(rej1) > args.max_chars:
                rej1 = rej1[:args.max_chars].rstrip()

            rec["rejected"] = rej1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Done. kept={kept}, dropped={dropped}, trimmed={trimmed}")
    print(f"Saved to: {args.out_path}")

if __name__ == "__main__":
    main()
