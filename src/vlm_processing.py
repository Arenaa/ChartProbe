"""Shared Hugging Face VLM input/output helpers (chat-template vs PaliGemma-style)."""

from __future__ import annotations

import ast
import io
import re
from dataclasses import dataclass

import torch

CHARTGEMMA_HUB_ID = "ahmed-masry/chartgemma"
CHARTGEMMA_PROMPT_PREFIX = "program of thought:"


def is_chartgemma_model(model_id: str) -> bool:
    mid = (model_id or "").strip().lower()
    return mid == CHARTGEMMA_HUB_ID or mid.endswith("/chartgemma")


def build_chartgemma_prompt(question: str) -> str:
    q = (question or "").strip()
    prefix = CHARTGEMMA_PROMPT_PREFIX
    if q.lower().startswith(prefix):
        return q
    return f"{prefix} {q}"


def encode_chartgemma_style(processor, image, prompt: str):
    """ChartGemma / PaliGemma: prepend image token; processor pairs image + text."""
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    text = build_chartgemma_prompt(prompt)
    if not text.startswith("<image>"):
        text = f"<image>{text}"
    return processor(text=text, images=image, return_tensors="pt")


@dataclass(frozen=True)
class ChartGemmaGenerationArgs:
    max_new_tokens: int


def chartgemma_generation_kwargs(gen: ChartGemmaGenerationArgs) -> dict:
    """Hub README recommends beam search (num_beams=4) for ChartGemma."""
    return {
        "max_new_tokens": gen.max_new_tokens,
        "num_beams": 4,
        "do_sample": False,
    }


def _format_pot_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif item is not None:
                parts.append(str(item))
        return ", ".join(parts)
    return str(value).strip()


def _pot_exec_env() -> tuple[dict, dict]:
    buf = io.StringIO()

    def capturing_print(*args, **kwargs) -> None:
        if kwargs.get("file") is not None:
            print(*args, **kwargs)
            return
        end = kwargs.get("end", "\n")
        sep = kwargs.get("sep", " ")
        text = sep.join(str(a) for a in args)
        buf.write(text)
        buf.write(end if isinstance(end, str) else "\n")

    safe_builtins = {
        "print": capturing_print,
        "len": len,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "tuple": tuple,
        "dict": dict,
        "set": set,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "sorted": sorted,
        "any": any,
        "all": all,
    }
    return safe_builtins, buf


def _normalize_pot_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    try:
        val = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text
    formatted = _format_pot_value(val)
    return formatted or text


def _execute_chartgemma_pot(code: str) -> str | None:
    code = (code or "").strip()
    if not code:
        return None
    env, buf = _pot_exec_env()
    try:
        exec(code, {"__builtins__": {}}, env)  # noqa: S102
    except Exception:
        return None
    out = buf.getvalue().strip()
    if not out:
        return None
    return _normalize_pot_text(out)


def _last_print_argument(code: str) -> str | None:
    code = (code or "").strip()
    if not code:
        return None
    matches = list(re.finditer(r"print\s*\(", code))
    if not matches:
        return None
    start = matches[-1].start()
    snippet = code[start:]
    try:
        node = ast.parse(snippet)
    except SyntaxError:
        return None
    for item in node.body:
        if isinstance(item, ast.Expr) and isinstance(item.value, ast.Call):
            call = item.value
            if isinstance(call.func, ast.Name) and call.func.id == "print" and call.args:
                try:
                    return _format_pot_value(ast.literal_eval(call.args[0]))
                except (ValueError, SyntaxError):
                    try:
                        return _format_pot_value(ast.unparse(call.args[0]))
                    except Exception:
                        return None
    return None


def _strip_print_wrapper(code: str) -> str:
    code = (code or "").strip()
    m = re.match(r"print\s*\((.*)\)\s*$", code, flags=re.DOTALL)
    if not m:
        return code
    inner = m.group(1).strip()
    if (inner.startswith('"') and inner.endswith('"')) or (inner.startswith("'") and inner.endswith("'")):
        try:
            return str(ast.literal_eval(inner))
        except (ValueError, SyntaxError):
            return inner.strip("'\"")
    return inner


def extract_chartgemma_pot_answer(prediction: str) -> str:
    """
    Turn ChartGemma program-of-thought output into text for RapidFuzz scoring.

    Order: execute PoT and capture stdout → parse last print(...) literal → strip print wrapper.
    """
    raw = (prediction or "").strip()
    if not raw:
        return ""

    executed = _execute_chartgemma_pot(raw)
    if executed:
        return executed

    parsed = _last_print_argument(raw)
    if parsed:
        return parsed

    if raw.startswith("print("):
        stripped = _strip_print_wrapper(raw)
        if stripped != raw:
            return _normalize_pot_text(stripped)

    # Multiline PoT: drop assignment lines, keep trailing expression/print lines.
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    kept = [ln for ln in lines if not re.match(r"^[A-Za-z_]\w*\s*=", ln)]
    if kept:
        joined = "\n".join(kept)
        executed = _execute_chartgemma_pot(joined)
        if executed:
            return executed
        parsed = _last_print_argument(joined)
        if parsed:
            return parsed
        if joined.startswith("print("):
            return _normalize_pot_text(_strip_print_wrapper(joined))
        return _normalize_pot_text(joined)

    return _normalize_pot_text(raw)


def processor_has_chat_template(processor) -> bool:
    if getattr(processor, "chat_template", None):
        return True
    tok = getattr(processor, "tokenizer", None)
    return bool(tok and getattr(tok, "chat_template", None))


def encode_messages_chat(processor, image, prompt: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def encode_paligemma_style(processor, image, prompt: str):
    """ChartGemma / PaliGemma: no chat template; see model card on the Hub."""
    if "<image>" not in prompt:
        prompt = f"<image>{prompt}"
    return processor(text=prompt, images=image, return_tensors="pt")


def move_inputs_to_device(
    inputs,
    device: torch.device,
    *,
    float_dtype: torch.dtype | None = None,
):
    """Move tensors to ``device``; optionally cast floating tensors (e.g. ``pixel_values``) to model dtype."""

    def _move_tensor(t: torch.Tensor) -> torch.Tensor:
        if t.is_floating_point() and float_dtype is not None:
            return t.to(device=device, dtype=float_dtype)
        return t.to(device)

    def _move_obj(obj: object) -> object:
        if torch.is_tensor(obj):
            return _move_tensor(obj)
        if isinstance(obj, dict):
            return {k: _move_obj(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = [_move_obj(x) for x in obj]
            return type(obj)(t)
        return obj

    # With dtype alignment, avoid BatchFeature.to(device) — it leaves floats as float32 on CUDA.
    if float_dtype is None and hasattr(inputs, "to") and callable(getattr(inputs, "to")):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return _move_obj(inputs)
    if hasattr(inputs, "keys") and callable(getattr(inputs, "keys", None)):
        return _move_obj(dict(inputs))
    return _move_obj(inputs)


def decode_generated(processor, output_ids: torch.Tensor, prompt_len: int, *, use_chat_template: bool) -> str:
    if use_chat_template:
        return processor.decode(output_ids[0, prompt_len:], skip_special_tokens=True).strip()
    texts = processor.batch_decode(
        output_ids[:, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return (texts[0] if texts else "").strip()


def patch_granite_create_causal_mask() -> None:
    """Granite Vision 4.1 remote code passes cache_position; transformers 5.9+ dropped it."""
    import transformers.masking_utils as masking_utils

    if getattr(masking_utils, "_chartnet_granite_mask_patched", False):
        return

    orig = masking_utils.create_causal_mask

    def _compat_create_causal_mask(*args, **kwargs):
        kwargs.pop("cache_position", None)
        if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        return orig(*args, **kwargs)

    masking_utils.create_causal_mask = _compat_create_causal_mask
    masking_utils._chartnet_granite_mask_patched = True


def prepare_vlm_inference(processor: object, model: object, *, model_id: str = "") -> None:
    """
    Normalize processor/model state for multimodal HF VLMs.

    Covers:
    - **LlavaProcessor**: ``patch_size`` / ``vision_feature_select_strategy`` missing from older processor JSON
      breaks token expansion (:class:`~transformers.models.llava.processing_llava.LlavaProcessor`).
    - **Granite Docling**: image preprocessor may reject ``resolution_max_side > max_image_size`` until limits are aligned.
    - **Generation**: ``model.config.eos_token_id`` / ``generation_config`` sometimes unset while the tokenizer is valid,
      which can break decoding paths across transformers revisions.
    """
    tok = getattr(processor, "tokenizer", None)
    cfg = getattr(model, "config", None)

    eos_id = getattr(tok, "eos_token_id", None)
    pad_id = getattr(tok, "pad_token_id", None)

    gc = getattr(model, "generation_config", None)
    if gc is not None and eos_id is not None and getattr(gc, "eos_token_id", None) in (None,):
        gc.eos_token_id = eos_id
    if gc is not None and pad_id is not None and getattr(gc, "pad_token_id", None) in (None,):
        gc.pad_token_id = pad_id

    if cfg is not None and eos_id is not None and getattr(cfg, "eos_token_id", None) in (None,):
        cfg.eos_token_id = eos_id

    tc = getattr(cfg, "text_config", None)
    if tc is not None and eos_id is not None and getattr(tc, "eos_token_id", None) in (None,):
        tc.eos_token_id = eos_id
    if tc is not None and pad_id is not None and getattr(tc, "pad_token_id", None) in (None,):
        tc.pad_token_id = pad_id

    proc_cls = type(processor).__name__
    if proc_cls.startswith("Llava") or "llava" in model_id.lower():
        vc = getattr(cfg, "vision_config", None)
        if cfg is not None and hasattr(processor, "patch_size") and getattr(processor, "patch_size") is None:
            ps = getattr(vc, "patch_size", None) if vc is not None else None
            if ps is not None:
                setattr(processor, "patch_size", int(ps))
        if cfg is not None and hasattr(processor, "vision_feature_select_strategy"):
            if getattr(processor, "vision_feature_select_strategy") is None:
                vfs = getattr(cfg, "vision_feature_select_strategy", None) or "default"
                setattr(processor, "vision_feature_select_strategy", vfs)

    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        rms = getattr(ip, "resolution_max_side", None)
        mis = getattr(ip, "max_image_size", None)
        try:
            if rms is not None and mis is not None:
                ri, mi = int(rms), int(mis)
                if ri > mi:
                    setattr(ip, "max_image_size", ri)
        except (TypeError, ValueError):
            pass
