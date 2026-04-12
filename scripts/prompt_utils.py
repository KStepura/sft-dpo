"""
Shared prompt formatting for SFT / DPO pair generation.

For Instruct-tuned models (e.g. Qwen2.5-Instruct), use prompt_style=chat so that
training data matches inference (chat template). Legacy Alpaca-style strings
remain available as prompt_style=alpaca.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def build_user_text(instruction: str, input_text: str) -> str:
    instr = (instruction or "").strip()
    inp = (input_text or "").strip()
    if inp:
        return f"{instr}\n\n{inp}"
    return instr


def format_alpaca_supervised(instruction: str, input_text: str, output: str) -> str:
    instr = (instruction or "").strip()
    inp = (input_text or "").strip()
    out = (output or "").strip()
    if inp:
        return (
            f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        )
    return f"### Instruction:\n{instr}\n\n### Response:\n{out}"


def format_alpaca_prompt_prefix(instruction: str, input_text: str) -> str:
    instr = (instruction or "").strip()
    inp = (input_text or "").strip()
    if inp:
        return f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n"
    return f"### Instruction:\n{instr}\n\n### Response:\n"


def format_chat_supervised(
    tok: PreTrainedTokenizerBase, instruction: str, input_text: str, output: str
) -> str:
    user = build_user_text(instruction, input_text)
    out = (output or "").strip()
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": user}, {"role": "assistant", "content": out}],
            tokenize=False,
            add_generation_prompt=False,
        )
    return format_alpaca_supervised(instruction, input_text, output)


def format_chat_prompt_prefix(
    tok: PreTrainedTokenizerBase, instruction: str, input_text: str
) -> str:
    user = build_user_text(instruction, input_text)
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return format_alpaca_prompt_prefix(instruction, input_text)
