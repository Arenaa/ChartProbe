"""
Chart QA CoT benchmark with RapidFuzz scoring (HuggingFace VLM backends).

Supports sweeps across HF presets / model ids (monolithic VLM checkpoints).

Evaluation protocol (ChartProbe paper):
- Pair each chart/scene image with its question (ChartProbe-9 or complex-reasoning).
- Prompt the model to output <think> and <answer> sections.
- Extract the final answer from <answer>.
- Score with type-aware metrics (9-Q) or RapidFuzz (complex reasoning).

Use --prompt-mode direct only for legacy direct-answer runs (no CoT XML).

Examples
--------
# ChartProbe / CR eval with a paper HF preset
python src/run_benchmark.py --preset internvl35-8b \
  --test-bundle data/bar_charts_500/eval_bundle --output-dir bench_runs/bar_internvl

# Multiple paper models
python src/run_benchmark.py \
  --preset internvl35-8b \
  --preset qwen35-9b \
  --preset llava-v16-mistral-7b

# Raw Hub ids
python src/run_benchmark.py \
  --model-id OpenGVLab/InternVL3_5-8B-HF \
  --model-id Qwen/Qwen3.5-9B

python src/run_benchmark.py --all-presets --continue-on-model-error
"""

from __future__ import annotations

import sys
from pathlib import Path

# Sibling modules live in src/; keep that directory on sys.path.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import argparse
import gc
import os
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from answer_scoring import aggregate_scores, fuzzy_ratio, score_answer
from question_parsing import parse_chartnet_question, parse_gold_thinking, parse_reference_fields
from vlm_processing import (
    ChartGemmaGenerationArgs,
    build_chartgemma_prompt,
    chartgemma_generation_kwargs,
    decode_generated,
    encode_chartgemma_style,
    encode_messages_chat,
    encode_paligemma_style,
    extract_chartgemma_pot_answer,
    is_chartgemma_model,
    move_inputs_to_device,
    patch_granite_create_causal_mask,
    prepare_vlm_inference,
    processor_has_chat_template,
)
from model_presets import MODEL_PRESETS, list_presets_text, resolve_model_id
from test_bundle import (
    default_test_bundle_dir,
    load_chartnet_samples,
    load_test_bundle,
    test_bundle_output_root,
)
from eval_metrics import attach_primary_metrics


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
for _name in ("httpx", "httpcore", "huggingface_hub"):
    logging.getLogger(_name).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _warn_transformers_vlms() -> None:
    """Qwen2.5-VL, Qwen3-VL, Qwen3.5, and related VLMs need recent transformers."""
    try:
        from packaging import version

        import transformers as tfm

        ver = version.parse(tfm.__version__)
        if ver < version.parse("4.57.0"):
            logger.warning(
                "transformers %s is older than 4.57; Qwen2.5-VL, Qwen3-VL, and several VLMs will not load. "
                "Run: pip install 'transformers>=4.57'",
                tfm.__version__,
            )
        elif ver < version.parse("5.0.0"):
            logger.warning(
                "transformers %s is older than 5.0; Qwen3.5-9B (qwen35-9b preset) requires transformers >= 5.0.",
                tfm.__version__,
            )
    except Exception:
        pass

COT_PROMPT_TEMPLATE = """You are given a chart image and a question about it.

Answer using chain-of-thought reasoning, then give the final answer.

Your reply MUST use exactly two XML-style sections in this order:

<think>
Your step-by-step reasoning about the chart.
</think>

<answer>
The final answer only (concise; match the style expected for chart QA).
</answer>

Question:
{question}"""

DIRECT_PROMPT_TEMPLATE = """You are given a chart image and a question about it.

Answer the question directly and concisely from the chart only. Do not show reasoning or chain-of-thought.

Question:
{question}"""

# Zero-training format sanity check: text-only exemplars teach the CoT XML /
# concise-answer convention without any weight update. Exemplars are probe-style
# (count / binary / short phrase) and do not include images — format only.
COT_FORMAT_ICL_PROMPT_TEMPLATE = """You are given a chart image and a question about it.

Answer using chain-of-thought reasoning, then give the final answer.

Your reply MUST use exactly two XML-style sections in this order:

<think>
Your step-by-step reasoning about the chart.
</think>

<answer>
The final answer only (concise; match the style expected for chart QA).
</answer>

Here are two examples of the expected reply format (different charts; format only):

Example 1 — question: How many bars appear to be above the average height?
<think>
Compare each bar height to the mean; count those strictly above it.
</think>
<answer>
3
</answer>

Example 2 — question: Is the largest bar more than twice the size of the smallest bar?
<think>
Read the max and min values from the chart and check whether max > 2 * min.
</think>
<answer>
Yes
</answer>

Now answer the following question about the provided chart image.

Question:
{question}"""


@dataclass(frozen=True)
class GenerationArgs:
    max_new_tokens: int
    do_sample: bool
    temperature: float
    repetition_penalty: float


@dataclass(frozen=True)
class Target:
    backend: str  # hf
    model_id: str
    run_tag: str
    display_name: str
    vision_note: str
    llm_note: str
    preset_name: str | None


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    return s.strip("_")[:120] or "run"


def _set_gen_seed(seed: int | None) -> None:
    if seed is None:
        return
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_cot_prompt(question: str) -> str:
    return COT_PROMPT_TEMPLATE.format(question=(question or "").strip())


def build_direct_prompt(question: str) -> str:
    return DIRECT_PROMPT_TEMPLATE.format(question=(question or "").strip())


def build_cot_format_icl_prompt(question: str) -> str:
    return COT_FORMAT_ICL_PROMPT_TEMPLATE.format(question=(question or "").strip())


def effective_prompt_mode(target: Target, requested: str) -> str:
    if requested == "cot" and (
        target.preset_name == "chartgemma" or is_chartgemma_model(target.model_id)
    ):
        return "chartgemma_pot"
    return requested


def build_prompt(
    question: str,
    prompt_mode: str = "cot",
    *,
    model_id: str | None = None,
    preset_name: str | None = None,
) -> str:
    if prompt_mode == "chartgemma_pot" or (
        prompt_mode == "cot"
        and (preset_name == "chartgemma" or is_chartgemma_model(model_id or ""))
    ):
        return build_chartgemma_prompt(question)
    if prompt_mode == "direct":
        return build_direct_prompt(question)
    if prompt_mode == "cot_format_icl":
        return build_cot_format_icl_prompt(question)
    return build_cot_prompt(question)


def extract_tag_body(text: str, tag: str) -> str:
    if not text:
        return ""
    pat = rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>"
    m = re.search(pat, text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _extract_malformed_answer(text: str) -> str:
    """Handle outputs missing ``<answer>`` but containing ``</answer>``."""
    text = (text or "").strip()
    if not text:
        return ""
    bodies = re.findall(
        r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE
    )
    if bodies:
        return bodies[-1].strip()
    m = re.search(r"</answer>", text, flags=re.IGNORECASE)
    if not m:
        return text
    before = text[: m.start()].strip()
    after = text[m.end() :].strip()
    if after:
        lines = [ln.strip() for ln in after.splitlines() if ln.strip()]
        return lines[-1] if lines else after
    lines = [ln.strip() for ln in before.splitlines() if ln.strip()]
    return lines[-1] if lines else before


def extract_model_answer(prediction: str, prompt_mode: str = "cot") -> str:
    if prediction is None:
        return ""
    if prompt_mode == "chartgemma_pot":
        return extract_chartgemma_pot_answer(prediction)
    if prompt_mode == "direct":
        return prediction.strip()
    # cot and cot_format_icl share <answer> extraction
    inner = extract_tag_body(prediction, "answer")
    if inner:
        return inner.strip()
    return _extract_malformed_answer(prediction)


def fuzzy_accuracy(pred: str, gold: str) -> float:
    return fuzzy_ratio(pred, gold)


def _run_eval_loop(
    *,
    samples: list[dict],
    target: Target,
    base_out: Path,
    predictor: Callable[[dict, str], str],
    prompt_mode: str = "cot",
    resume: bool = False,
    eval_metadata: dict | None = None,
) -> Path:
    out_dir = base_out / "runs" / _slug(target.run_tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "cot_fuzz_results.jsonl"
    metrics_path = out_dir / "cot_fuzz_metrics.json"

    scores: list[float] = []
    score_rows: list[dict] = []
    latencies: list[float] = []
    completed_indices: set[int] = set()

    if resume and results_path.is_file():
        with open(results_path, encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                idx = int(row["index"])
                completed_indices.add(idx)
                scores.append(float(row.get("fuzzy_accuracy", 0.0)))
                latencies.append(float(row.get("latency_s", 0.0)))
                score_rows.append(row)
        logger.info(
            "Resuming %s: %d/%d samples already complete",
            target.run_tag,
            len(completed_indices),
            len(samples),
        )
    else:
        results_path.unlink(missing_ok=True)

    logger.info(
        "Running %s on %d samples [prompt_mode=%s resume=%s] -> %s",
        target.display_name,
        len(samples),
        prompt_mode,
        resume,
        out_dir,
    )

    for i, sample in enumerate(samples):
        if i in completed_indices:
            continue
        t0 = time.time()
        q_text = parse_chartnet_question(sample.get("complex_question", ""))
        prompt_for_model = build_prompt(
            q_text,
            prompt_mode,
            model_id=target.model_id,
            preset_name=target.preset_name,
        )
        gold_answer, ref_raw = parse_reference_fields(sample.get("full_reasoning", ""))
        gold_thinking = parse_gold_thinking(sample.get("full_reasoning", ""))
        answer_type = sample.get("answer_type")

        prediction_error: str | None = None
        try:
            prediction = predictor(sample, prompt_for_model)
        except Exception as exc:
            prediction_error = f"{type(exc).__name__}: {exc}"
            logger.error("Sample %d failed (%s): %s", i, target.display_name, prediction_error)
            prediction = ""

        extracted = extract_model_answer(prediction, prompt_mode)
        prediction_thinking = (
            extract_tag_body(prediction, "redacted_thinking")
            if prompt_mode in ("cot", "cot_format_icl")
            else ""
        )
        scored = score_answer(extracted, gold_answer, answer_type=answer_type)
        for meta_key in ("stage", "template_id", "answer_type", "chart_id"):
            if sample.get(meta_key) is not None:
                scored[meta_key] = sample.get(meta_key)
        acc = float(scored["fuzzy_accuracy"])
        latency = time.time() - t0
        scores.append(acc)
        score_rows.append(scored)
        latencies.append(latency)

        row = {
            "index": i,
            "id": sample.get("id", ""),
            "question": q_text,
            "prompt_for_model": prompt_for_model,
            "prediction_raw": prediction,
            "prediction_thinking": prediction_thinking,
            "prediction_answer_extracted": extracted,
            "gold_reference_answer": gold_answer,
            "gold_thinking": gold_thinking,
            "fuzzy_accuracy": scored["fuzzy_accuracy"],
            "latency_s": round(latency, 3),
            "reference_raw": ref_raw,
            "prediction_error": prediction_error,
            "backend": target.backend,
            "model_id": target.model_id,
            "preset": target.preset_name,
        }
        for key in (
            "fuzzy_token_sort_ratio",
            "exact_match",
            "exact_label_match",
            "fuzzy_ge_80",
            "fuzzy_ge_90",
            "type_aware_correct",
            "hard_correct",
            "binary_match",
            "numeric_exact_match",
            "numeric_tol_match",
            "answer_type",
        ):
            if key in scored:
                row[key] = scored[key]
        for meta in ("stage", "template_id", "chart_id", "complexity_bin", "bar_count"):
            if sample.get(meta) is not None:
                row[meta] = sample.get(meta)
        with open(results_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

        if (i + 1) % 10 == 0:
            logger.info(
                "[%s %d/%d] mean fuzzy acc: %.4f | mean latency: %.2fs",
                target.run_tag,
                i + 1,
                len(samples),
                sum(scores) / len(scores),
                sum(latencies) / len(latencies),
            )

    mean_acc = sum(scores) / len(scores) if scores else 0.0
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    if prompt_mode == "chartgemma_pot":
        benchmark_name = "chart_qa_chartgemma_pot_rapidfuzz"
        eval_protocol = (
            "Pair chart image + complex_question; ChartGemma program-of-thought prompt "
            "(see ahmed-masry/chartgemma); execute/parse PoT stdout for scoring"
        )
    elif prompt_mode == "direct":
        benchmark_name = "chart_qa_direct_rapidfuzz"
        eval_protocol = (
            "Pair chart image + complex_question; direct-answer prompt; "
            "score extracted answer vs gold"
        )
    else:
        benchmark_name = "chart_qa_cot_think_answer_rapidfuzz"
        eval_protocol = (
            "Pair chart image + complex_question; prompt for "
            "<think>/<answer>; score extracted <answer> vs gold"
        )
    metrics = {
        "benchmark": benchmark_name,
        "eval_protocol": eval_protocol,
        "prompt_mode": prompt_mode,
        "backend": target.backend,
        "model_id": target.model_id,
        "preset": target.preset_name,
        "run_tag": target.run_tag,
        "display_name": target.display_name,
        "vision_encoder_note": target.vision_note,
        "llm_note": target.llm_note,
        "num_samples": len(samples),
        "avg_fuzzy_accuracy": round(mean_acc, 6),
        "avg_fuzzy_accuracy_percent": round(mean_acc * 100.0, 4),
        "avg_latency_s": round(mean_latency, 4),
        "rapidfuzz_scorer": "fuzz.ratio(extracted_answer, gold_answer) / 100",
        "gold_field": "full_reasoning JSON answer (fallback think per parse_reference_fields)",
        "reasoning_traces_captured": prompt_mode == "cot",
        "trace_fields": (
            ["prediction_raw", "prediction_thinking", "gold_thinking", "reference_raw"]
            if prompt_mode == "cot"
            else []
        ),
        "scoring_notes": (
            "Additional metrics: exact_match, fuzzy_ge_80/90, type_aware_correct "
            "(uses answer_type when present), binary_match, numeric_exact_match, numeric_tol_match"
        ),
    }
    metrics.update(aggregate_scores(score_rows))
    if eval_metadata:
        metrics.update(eval_metadata)
    metrics = attach_primary_metrics(
        metrics,
        run_tag=target.run_tag,
        output_dir=base_out,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Done %s | avg fuzzy acc %.4f | avg lat %.3fs", target.run_tag, mean_acc, mean_latency)
    return results_path


def _normalize_model_id_arg(model_id: str) -> str:
    """
    If ``model_id`` points at an existing directory, return its resolved absolute path so
    ``transformers`` loads locally. Otherwise leave Hub ids unchanged.

    Passing a missing path triggers a clear error (HF Hub otherwise treats strings like
    ``/workspace/foo`` as invalid repo IDs).
    """
    raw = model_id.strip()
    p = Path(raw).expanduser()
    looks_like_path = p.is_absolute() or raw.startswith(("./", "../"))
    if not looks_like_path:
        return raw
    if not p.is_dir():
        raise FileNotFoundError(
            f"HF model path is not an existing directory: {raw}. "
            f"Either fix --model-id / CHARTNET_HF_* or use a Hub id "
            f"(e.g. ahmed-masry/chartgemma)."
        )
    return str(p.resolve(strict=False))


def _build_vlm_predictor(
    model_id: str,
    gen: GenerationArgs,
    *,
    trust_remote_code: bool,
    torch_dtype: str,
    device_map: str,
    hub_token: str | None,
    cuda_device: int | None = None,
    lora_adapter: str | None = None,
):
    model_id = _normalize_model_id_arg(model_id)
    if "granite" in model_id.lower():
        patch_granite_create_causal_mask()
    dtype = getattr(torch, torch_dtype)
    processor_path = lora_adapter if lora_adapter and Path(lora_adapter).is_dir() else model_id
    parent = Path(lora_adapter).parent if lora_adapter else None
    if parent and (parent / "processor").is_dir():
        processor_path = str(parent / "processor")
    processor = AutoProcessor.from_pretrained(
        processor_path, trust_remote_code=trust_remote_code, token=hub_token
    )
    load_kw: dict[str, Any] = dict(
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
        token=hub_token,
    )
    dm = (device_map or "").strip().lower()
    try:
        if dm in ("none", "off", "false", "0"):
            # Whole model on one device; no Hugging Face / Accelerate device_map (avoids accelerate import).
            model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kw)
            if torch.cuda.is_available():
                idx = 0 if cuda_device is None else int(cuda_device)
                model = model.to(torch.device(f"cuda:{idx}"))
            else:
                model = model.to(torch.device("cpu"))
        else:
            model = AutoModelForImageTextToText.from_pretrained(model_id, device_map=device_map, **load_kw)
    except (ValueError, KeyError) as e:
        err = str(e).lower()
        if (
            "qwen2_5_vl" in err
            or "qwen3_vl" in err
            or "qwen3_5" in err
            or ("model type" in err and "qwen" in err)
            or "does not recognize this architecture" in err
        ):
            raise RuntimeError(
                f"Loading {model_id} failed: {e!s}. Upgrade transformers to >=4.57."
            ) from e
        raise
    model.eval()

    if lora_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_adapter, is_trainable=False)
        model.eval()
        logger.info("Loaded LoRA adapter from %s", lora_adapter)

    prepare_vlm_inference(processor, model, model_id=model_id)

    chartgemma = is_chartgemma_model(model_id)
    use_chat_template = processor_has_chat_template(processor)

    def predict(sample: dict, prompt_for_model: str) -> str:
        image = sample["image"]
        if chartgemma:
            inputs = encode_chartgemma_style(processor, image, prompt_for_model)
        elif use_chat_template:
            inputs = encode_messages_chat(processor, image, prompt_for_model)
        else:
            inputs = encode_paligemma_style(processor, image, prompt_for_model)
        param = next(model.parameters(), None)
        in_dtype = param.dtype if param is not None and param.is_floating_point() else None
        inputs = move_inputs_to_device(inputs, model.device, float_dtype=in_dtype)

        if chartgemma:
            kwargs = chartgemma_generation_kwargs(
                ChartGemmaGenerationArgs(max_new_tokens=gen.max_new_tokens)
            )
        else:
            kwargs = {
                "max_new_tokens": gen.max_new_tokens,
                "do_sample": gen.do_sample,
            }
            if gen.do_sample:
                kwargs["temperature"] = gen.temperature
            if gen.repetition_penalty > 0:
                kwargs["repetition_penalty"] = gen.repetition_penalty

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **kwargs)

        prompt_len = int(inputs["input_ids"].shape[1])
        return decode_generated(
            processor, output_ids, prompt_len, use_chat_template=use_chat_template
        )

    return {"processor": processor, "model": model, "predict": predict}


def _cleanup_runtime(obj: dict[str, Any]) -> None:
    for k in list(obj.keys()):
        try:
            del obj[k]
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_targets(args: argparse.Namespace) -> list[Target]:
    targets: list[Target] = []

    hf_specs: list[tuple[str, str]] = []
    if args.preset:
        for p in args.preset:
            hf_specs.append(("preset", p))
    if args.model_ids:
        for m in args.model_ids:
            hf_specs.append(("model", m))
    if args.all_presets:
        for p in sorted(MODEL_PRESETS.keys()):
            hf_specs.append(("preset", p))

    for typ, value in hf_specs:
        if typ == "preset":
            model_id, meta = resolve_model_id(preset=value, model_id=None)
            preset_name = value
            disp = f"hf:{value}"
        else:
            model_id, meta = resolve_model_id(preset=None, model_id=value)
            preset_name = None
            disp = f"hf:{value}"

        run_tag = _slug(f"cotfuzz_hf_{preset_name or model_id}")
        targets.append(
            Target(
                backend="hf",
                model_id=model_id,
                run_tag=run_tag,
                display_name=disp,
                vision_note=(meta or {}).get("vision_encoder", ""),
                llm_note=(meta or {}).get("llm_params_note", ""),
                preset_name=preset_name,
            )
        )

    if args.run_tag:
        if len(targets) != 1:
            raise ValueError("--run-tag can only be used when exactly one target model is selected.")
        t = targets[0]
        targets[0] = Target(
            backend=t.backend,
            model_id=t.model_id,
            run_tag=args.run_tag,
            display_name=t.display_name,
            vision_note=t.vision_note,
            llm_note=t.llm_note,
            preset_name=t.preset_name,
        )

    return targets


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Chart QA CoT + RapidFuzz benchmark across HuggingFace VLM models."
    )
    ap.add_argument("--list-presets", action="store_true")

    ap.add_argument("--backend", choices=("hf",), default="hf", help="Model backend (HF only).")

    ap.add_argument("--preset", action="append", default=None, help="Model preset key (repeatable).")
    ap.add_argument("--model-id", dest="model_ids", action="append", default=None, help="Hub model id (repeatable).")
    ap.add_argument("--all-presets", action="store_true", help="Run all known model presets.")

    ap.add_argument("--run-tag", type=str, default=None, help="Custom run tag (single target only).")
    ap.add_argument(
        "--test-bundle",
        type=Path,
        default=default_test_bundle_dir(REPO),
        help="Bundle dir or samples.jsonl (bar_chart_bundle/, bar_chart_data/, optional test_bundle/). "
        "Default: bar_chart_bundle/ when present, else first sibling dir with samples.jsonl. "
        "Ignored when --hf-config is set.",
    )
    ap.add_argument(
        "--hf-config",
        choices=("human_verified", "reasoning"),
        default=None,
        help=(
            "Load eval samples directly from HuggingFace instead of --test-bundle. "
            "human_verified/test = official 2,000-chart held-out set (no QA). "
            "reasoning/train = complex-reasoning QA with CoT gold (use --prompt-mode cot)."
        ),
    )
    ap.add_argument(
        "--hf-split",
        type=str,
        default=None,
        help="HF split name (default: test for human_verified, train for reasoning).",
    )
    ap.add_argument(
        "--hf-sample-seed",
        type=int,
        default=42,
        help="Random seed when subsampling reasoning rows (--max-samples with --hf-config reasoning).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Base directory for runs/ subfolders (default: test-bundle root). "
            "If omitted, CHARTNET_OUTPUT_DIR is used when set (for launchers that avoid CLI passthrough)."
        ),
    )
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run: skip samples already in cot_fuzz_results.jsonl.",
    )

    ap.add_argument(
        "--prompt-mode",
        choices=("cot", "direct", "cot_format_icl"),
        default="cot",
        help=(
            "'cot' (default): prompt for <think>/<answer>, extract <answer>, "
            "score with RapidFuzz vs gold. "
            "'direct': legacy direct-answer prompt (no CoT XML). "
            "'cot_format_icl': zero-train format sanity — same CoT XML plus two "
            "text-only probe-style exemplars (no weight update)."
        ),
    )
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--repetition-penalty", type=float, default=1.12)
    ap.add_argument(
        "--gen-seed",
        type=int,
        default=None,
        help="Seed for torch/random/numpy before generation (use with --do-sample for repeatable stochastic runs).",
    )

    ap.add_argument("--torch-dtype", type=str, default="bfloat16")
    ap.add_argument(
        "--device-map",
        type=str,
        default="none",
        help=(
            'HF load device_map. Default "none" loads the full model on one GPU (.to(cuda:N)) '
            "and avoids split-model bugs (cuda:0 vs cuda:1) on multi-GPU hosts. "
            'Use "auto" only if you need model parallelism across GPUs.'
        ),
    )
    ap.add_argument("--no-trust-remote-code", action="store_true")
    ap.add_argument("--hub-token", type=str, default=None, help="Hugging Face Hub token for gated models.")
    ap.add_argument(
        "--lora-adapter",
        type=str,
        default=None,
        help="Path to PEFT LoRA adapter directory (from finetune_lora.py).",
    )

    ap.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="CUDA device index for this process. Ignored when using launcher with CUDA_VISIBLE_DEVICES.",
    )
    ap.add_argument(
        "--continue-on-model-error",
        action="store_true",
        help=(
            "If loading or eval fails for one target, log the error and continue with the rest. "
            "Exit code 1 if any target failed. Useful for large sweeps when some checkpoints are gated."
        ),
    )

    args = ap.parse_args()

    if args.output_dir is None:
        _out_env = os.environ.get("CHARTNET_OUTPUT_DIR")
        if _out_env:
            args.output_dir = Path(_out_env)

    if args.list_presets:
        print(list_presets_text())
        return

    bundle = args.test_bundle.resolve()
    if args.hf_config:
        hf_split = args.hf_split
        if hf_split is None:
            hf_split = "test" if args.hf_config == "human_verified" else "train"
        samples = load_chartnet_samples(
            hf_config=args.hf_config,
            hf_split=hf_split,
            max_samples=args.max_samples,
            seed=args.hf_sample_seed,
        )
        if args.hf_config == "human_verified" and args.prompt_mode == "cot":
            raise ValueError(
                "human_verified splits have no complex_question/full_reasoning QA gold. "
                "For CoT + RapidFuzz complex-reasoning eval use --hf-config reasoning "
                "(or --test-bundle with a reasoning export such as eval_suite_reasoning_2000/)."
            )
        bundle_root = (args.output_dir or REPO / f"bench_runs/hf_{args.hf_config}_{hf_split}").expanduser().resolve()
    else:
        samples = load_test_bundle(bundle)
        bundle_root = (args.output_dir or test_bundle_output_root(args.test_bundle)).expanduser().resolve()
        if args.max_samples is not None:
            samples = samples[: max(0, args.max_samples)]

    targets = _resolve_targets(args)
    if not targets:
        raise ValueError(
            "No target models selected. Use --preset / --model-id / --all-presets."
        )

    logger.info("Selected %d target(s).", len(targets))
    for t in targets:
        logger.info("  - %s -> %s", t.run_tag, t.model_id)

    _warn_transformers_vlms()

    _set_gen_seed(args.gen_seed)

    gen = GenerationArgs(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
    )
    eval_metadata = {
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else 0.0,
        "gen_seed": args.gen_seed,
    }

    hub_token = (
        args.hub_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not hub_token:
        logger.warning("No HF token set; gated checkpoints may fail (HF_TOKEN / HUGGING_FACE_HUB_TOKEN).")

    base_out = bundle_root
    logger.info("Benchmark outputs → %s/runs/<run_tag>/", base_out)

    failed_run_tags: list[str] = []
    for idx, target in enumerate(targets, start=1):
        logger.info("[%d/%d] Start %s", idx, len(targets), target.display_name)

        runtime: dict[str, Any] = {}
        try:
            runtime = _build_vlm_predictor(
                target.model_id,
                gen,
                trust_remote_code=not args.no_trust_remote_code,
                torch_dtype=args.torch_dtype,
                device_map=args.device_map,
                hub_token=hub_token,
                cuda_device=args.cuda_device,
                lora_adapter=args.lora_adapter,
            )
            predict = runtime["predict"]

            prompt_mode = effective_prompt_mode(target, args.prompt_mode)
            if prompt_mode != args.prompt_mode:
                logger.info(
                    "Using prompt_mode=%s for %s (ChartGemma does not follow CoT XML prompts).",
                    prompt_mode,
                    target.display_name,
                )
            _run_eval_loop(
                samples=samples,
                target=target,
                base_out=base_out,
                predictor=predict,
                prompt_mode=prompt_mode,
                resume=args.resume,
                eval_metadata=eval_metadata,
            )
        except Exception as e:
            logger.exception("[%d/%d] FAILED %s (%s)", idx, len(targets), target.display_name, target.run_tag)
            if args.continue_on_model_error:
                failed_run_tags.append(target.run_tag)
                continue
            raise
        finally:
            _cleanup_runtime(runtime)

    if failed_run_tags:
        logger.error(
            "Finished with %d failed target(s): %s",
            len(failed_run_tags),
            ", ".join(failed_run_tags),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
