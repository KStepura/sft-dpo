import argparse, json, os, re

BAD_MARKERS = [
    "### Instruction:",
    "Human:",
    "User:",
    "Assistant:",
]

def strip_at_marker(text: str) -> str:
    if text is None:
        return ""
    t = text.strip()
    # режем на первом появлении любого маркера (кроме начала строки)
    for m in BAD_MARKERS:
        idx = t.find(m)
        if idx > 0:
            t = t[:idx].strip()
    return t

def has_bad_marker_inside(text: str) -> bool:
    if not text:
        return False
    # если маркер встречается НЕ в начале — это склейка
    for m in BAD_MARKERS:
        idx = text.find(m)
        if idx > 0:
            return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--min_chars", type=int, default=30)
    ap.add_argument("--drop_if_marker_inside", action="store_true")
    args = ap.parse_args()

    kept = dropped = trimmed = 0
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    with open(args.in_path, "r", encoding="utf-8") as fin, open(args.out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)

            prompt = (r.get("prompt") or "").strip()
            chosen0 = (r.get("chosen") or "")
            rejected0 = (r.get("rejected") or "")

            chosen = strip_at_marker(chosen0)
            rejected = strip_at_marker(rejected0)

            if chosen != chosen0.strip() or rejected != rejected0.strip():
                trimmed += 1

            if args.drop_if_marker_inside:
                if has_bad_marker_inside(chosen) or has_bad_marker_inside(rejected):
                    dropped += 1
                    continue

            if len(prompt) < 10 or len(chosen) < args.min_chars or len(rejected) < args.min_chars:
                dropped += 1
                continue

            # basic sanity: rejected != chosen
            if chosen.strip() == rejected.strip():
                dropped += 1
                continue

            fout.write(json.dumps({"prompt": prompt, "chosen": chosen, "rejected": rejected}, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Done. kept={kept}, dropped={dropped}, trimmed={trimmed}")
    print("Saved to:", args.out_path)

if __name__ == "__main__":
    main()
