import re
import json
import argparse
from collections import Counter

# режем любые "переходы" к следующему диалогу/инструкции
CUT_PATTERNS_ALPACA = [
    r"###\s*Instruction\s*:",
    r"###\s*Input\s*:",
    r"###\s*Response\s*:",
    r"\bHuman\s*:",
    r"\bUser\s*:",
    r"\bAssistant\s*:",
    r"You are an AI assistant\.",
]

# Qwen-style chat markers (and generic role tags)
CUT_PATTERNS_CHAT = CUT_PATTERNS_ALPACA + [
    r"<\|im_start\|>\s*user",
    r"<\|im_start\|>\s*system",
    r"<\|im_start\|>\s*assistant",
    r"<\|im_end\|>",
    r"<\|redacted_im_end\|>",
]


def norm_ws(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def clean_prompt(p: str, prompt_style: str) -> str:
    p = norm_ws(p)
    if prompt_style == "alpaca":
        if "### Response:" in p:
            head, _sep, _tail = p.partition("### Response:")
            p = head + "### Response:"
        else:
            p = p + "\n\n### Response:"
    return p


def cut_at_markers(x: str, patterns: list[str]) -> str:
    """Обрезает по самому раннему вхождению любого маркера (даже если он не на новой строке)."""
    if not x:
        return ""
    hits = []
    for pat in patterns:
        m = re.search(pat, x)
        if m:
            hits.append(m.start())
    if hits:
        x = x[:min(hits)].strip()
    return x


def postprocess_text(x: str, prompt_style: str) -> str:
    x0 = x or ""
    x = norm_ws(x0)

    # если модель начала с "### Response:" — убираем
    x = re.sub(r"^\s*###\s*Response\s*:\s*", "", x).strip()

    patterns = CUT_PATTERNS_CHAT if prompt_style == "chat" else CUT_PATTERNS_ALPACA
    x = cut_at_markers(x, patterns)

    # иногда "### Response:" встречается несколько раз — оставим последнюю часть
    # (после cut_at_markers это обычно уже не нужно, но оставим на всякий)
    resp = "### Response:"
    if x.count(resp) >= 2:
        x = x.split(resp)[-1].strip()

    # мягко лечим незакрытые ``` (если нечётное число)
    if x.count("```") % 2 == 1:
        x = x.rsplit("```", 1)[0].strip()

    return x


def looks_truncated(x: str) -> bool:
    x = (x or "").strip()
    if not x:
        return True
    if x.endswith(("...", "…", ":", "-", "—", ",")):
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--min_chars", type=int, default=20)
    ap.add_argument("--drop_if_truncated", action="store_true")
    ap.add_argument("--drop_if_equal", action="store_true")
    ap.add_argument(
        "--prompt_style",
        type=str,
        choices=("chat", "alpaca"),
        default="chat",
        help="Должен совпадать с build_dpo_pairs / train_sft.",
    )
    ap.add_argument(
        "--make_strict_prompt",
        action="store_true",
        help="alpaca: prompt заканчивается на '### Response:'. chat: убирает лишние хвостовые \\n.",
    )
    ap.add_argument("--stats_path", default=None, help="куда сохранить статистику (json)")
    args = ap.parse_args()

    kept = 0
    dropped = 0
    trimmed = 0
    reasons = Counter()

    with open(args.in_path, "r", encoding="utf-8") as fin, open(args.out_path, "w", encoding="utf-8") as fout:
        for ln, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                dropped += 1
                reasons["bad_json"] += 1
                continue

            prompt = obj.get("prompt", "")
            chosen = obj.get("chosen", "")
            rejected = obj.get("rejected", "")

            prompt2 = clean_prompt(prompt, args.prompt_style)
            chosen2 = postprocess_text(chosen, args.prompt_style)
            rejected2 = postprocess_text(rejected, args.prompt_style)

            if (prompt2 != norm_ws(prompt)) or (chosen2 != norm_ws(chosen)) or (rejected2 != norm_ws(rejected)):
                trimmed += 1

            if len(chosen2) < args.min_chars:
                dropped += 1
                reasons["chosen_too_short"] += 1
                continue
            if len(rejected2) < args.min_chars:
                dropped += 1
                reasons["rejected_too_short"] += 1
                continue

            if args.drop_if_truncated and (looks_truncated(chosen2) or looks_truncated(rejected2)):
                dropped += 1
                reasons["looks_truncated"] += 1
                continue

            if args.drop_if_equal and chosen2 == rejected2:
                dropped += 1
                reasons["chosen_eq_rejected"] += 1
                continue

            if args.make_strict_prompt:
                if args.prompt_style == "alpaca":
                    out_prompt = prompt2
                else:
                    out_prompt = prompt2.rstrip("\n")
            else:
                out_prompt = prompt2 + "\n"

            out = {
                "prompt": out_prompt,
                "chosen": chosen2,
                "rejected": rejected2,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Done. kept={kept}, dropped={dropped}, trimmed={trimmed}")
    print(f"Saved to: {args.out_path}")

    if args.stats_path:
        stats = {
            "kept": kept,
            "dropped": dropped,
            "trimmed": trimmed,
            "reasons": dict(reasons),
            "in_path": args.in_path,
            "out_path": args.out_path,
        }
        with open(args.stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"Stats saved to: {args.stats_path}")


if __name__ == "__main__":
    main()
