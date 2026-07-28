"""
Named VLM checkpoints for ChartNet experiments.

Each preset is one Hugging Face **monolithic** model: vision encoder + projector + LLM
are trained and released together. “Different vision encoder / size” is achieved by
choosing a different preset (or any `--model-id`), not by swapping modules at runtime.

All presets below use the standard HF path: AutoProcessor + AutoModelForImageTextToText
with trust_remote_code=True (required for InternVL-style remote modeling code).

Verify IDs and licenses on the Hub before running; add your own presets to MODEL_PRESETS.

Curated HF presets include: SmolVLM-256M-Instruct, IBM Granite Docling / Vision 3.3 / 4.x,
Qwen2.5-VL-3B, LLaVA-1.6-Mistral-7B, Qwen3-VL-4B-Instruct (no official Qwen3-VL-3B on the Hub),
InternVL3.5-8-HF, ChartGemma, plus optional smaller/larger InternVL3.5 sizes.
"""

from __future__ import annotations

from typing import TypedDict


class ModelPreset(TypedDict, total=False):
    model_id: str
    vision_encoder: str
    llm_params_note: str
    notes: str
    reasoning_prompt_mode: str


# Presets are grouped roughly by scale / role; extra InternVL3.5 sizes remain for sweeps.
MODEL_PRESETS: dict[str, ModelPreset] = {
    # --- Compact / “on the shelf” small VLMs ---
    "smolvlm-instruct": {
        "model_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "vision_encoder": "SmolVLM vision @ ~512px (see card)",
        "llm_params_note": "~256M total (SmolVLM family)",
        "notes": "Alias for smolvlm-256m-instruct (256M Instruct checkpoint).",
    },
    "smolvlm-256m-instruct": {
        "model_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "vision_encoder": "SmolVLM vision @ ~512px (see card)",
        "llm_params_note": "~256M total (SmolVLM family)",
        "notes": "For SmolVLM2 line use e.g. HuggingFaceTB/SmolVLM2-256M-Instruct via --model-id.",
    },
    "granite-docling-258m": {
        "model_id": "ibm-granite/granite-docling-258m",
        "vision_encoder": "Docling multimodal encoder (see IBM card)",
        "llm_params_note": "~258M",
    },
    "granite-vision-3.3-2b": {
        "model_id": "ibm-granite/granite-vision-3.3-2b",
        "vision_encoder": "Granite Vision 3.3 stack (see IBM card)",
        "llm_params_note": "~2B",
    },
    "granite-4.0-3b-vision": {
        "model_id": "ibm-granite/granite-4.0-3b-vision",
        "vision_encoder": "Granite 4.0 vision stack (see IBM card)",
        "llm_params_note": "~3B",
    },
    "granite-vision-4.1-4b": {
        "model_id": "ibm-granite/granite-vision-4.1-4b",
        "vision_encoder": "Granite Vision 4.1 stack (see IBM card)",
        "llm_params_note": "~4B (3.4B LLM + 0.6B vision; LoRA on Granite-4.1-3B)",
        "notes": (
            "Chart VLM for stage LoRA + complex-reasoning eval. "
            "ibm-granite/granite-4.1-3b is the text-only LM backbone and cannot train on chart images; "
            "use this vision checkpoint instead."
        ),
    },
    # Alias: users often ask for granite-4.1-3b; chart recipes must use the vision checkpoint.
    "granite-4.1-3b": {
        "model_id": "ibm-granite/granite-vision-4.1-4b",
        "vision_encoder": "Granite Vision 4.1 stack (see IBM card)",
        "llm_params_note": "~4B VLM (alias of granite-vision-4.1-4b)",
        "notes": (
            "Alias for granite-vision-4.1-4b. The Hub id ibm-granite/granite-4.1-3b is text-only; "
            "this preset loads the vision model built on that LM for chart fine-tune/eval."
        ),
    },
    "qwen25-vl-3b-instruct": {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "vision_encoder": "Qwen2.5-VL ViT (dynamic resolution)",
        "llm_params_note": "~3B instruct",
    },
    "llava-v16-mistral-7b": {
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "vision_encoder": "CLIP ViT-L (LLaVA-1.6)",
        "llm_params_note": "Mistral-7B backbone",
    },
    "llava-onevision-7b": {
        "model_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "vision_encoder": "SigLIP SO400M (LLaVA-OneVision anyres)",
        "llm_params_note": "Qwen2-7B backbone (~7B)",
        "notes": "LlavaOnevisionForConditionalGeneration via AutoModelForImageTextToText.",
    },
    # --- Larger / gated / chart-focused (same AutoModelForImageTextToText path where supported) ---
    "qwen3-vl-4b-instruct": {
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "vision_encoder": "Qwen3-VL ViT (see Hub card)",
        "llm_params_note": "~4B instruct",
        "notes": "There is no Qwen3-VL-3B-Instruct on the Hub; 4B is the closest mid-size instruct model (2B also exists as Qwen/Qwen3-VL-2B-Instruct).",
    },
    "qwen35-9b": {
        "model_id": "Qwen/Qwen3.5-9B",
        "vision_encoder": "Qwen3.5 ViT (early-fusion multimodal; see Hub card)",
        "llm_params_note": "~9B instruct",
        "notes": "Requires transformers >= 5.0 (Qwen3_5 architecture).",
    },
    "internvl35-2b": {
        "model_id": "OpenGVLab/InternVL3_5-2B-HF",
        "vision_encoder": "InternViT-style ViT (~150M class, 448px; see card)",
        "llm_params_note": "~2B effective (dense small)",
    },
    "internvl35-8b": {
        "model_id": "OpenGVLab/InternVL3_5-8B-HF",
        "vision_encoder": "InternViT-style ViT (~150M class, 448px; see card)",
        "llm_params_note": "~8B language backbone (dense)",
    },
    "internvl35-14b": {
        "model_id": "OpenGVLab/InternVL3_5-14B-HF",
        "vision_encoder": "Stronger ViT stack vs2B/8B (see Hub card)",
        "llm_params_note": "~14B language backbone",
    },
    "chartgemma": {
        "model_id": "ahmed-masry/chartgemma",
        "vision_encoder": "SigLIP-style + PaliGemma backbone (see model card)",
        "llm_params_note": "Chart instruction-tuned PaliGemma-class VLM",
        "notes": (
            "ChartGemma / ahmed-masry/chartgemma on the Hub (chart QA / reasoning). "
            "Baseline-only in the bar/pie CR500 recipe (PoT prompt)."
        ),
        "reasoning_prompt_mode": "chartgemma_pot",
    },
    "paligemma-3b-pt-448": {
        "model_id": "google/paligemma-3b-pt-448",
        "vision_encoder": "SigLIP So400m/14 @ 448px (PaliGemma)",
        "llm_params_note": "~3B pretrained (Gemma-2B + SigLIP)",
        "notes": (
            "Gated Hub model — requires HF login / HF_TOKEN. "
            "Stage LoRA fine-tune + complex-reasoning eval (bar/pie 500)."
        ),
    },
}


def list_presets_text() -> str:
    lines = ["Presets (--preset NAME):\n"]
    for name, p in sorted(MODEL_PRESETS.items()):
        lines.append(f"  {name}")
        lines.append(f"      model_id: {p['model_id']}")
        lines.append(f"      vision:   {p.get('vision_encoder', '')}")
        lines.append(f"      llm:      {p.get('llm_params_note', '')}")
        if p.get("notes"):
            lines.append(f"      notes:    {p['notes']}")
        lines.append("")
    lines.append("Or pass any compatible Hub id with --model-id (overrides --preset).")
    return "\n".join(lines)


def resolve_model_id(*, preset: str | None, model_id: str | None, default_preset: str = "internvl35-8b") -> tuple[str, ModelPreset | None]:
    """
    Returns (hub_model_id, preset_dict_or_none).
    If model_id is set, it wins and preset metadata is None unless model_id matches a preset.
    """
    if model_id and model_id.strip():
        mid = model_id.strip()
        for _, meta in MODEL_PRESETS.items():
            if meta["model_id"] == mid:
                return mid, meta
        return mid, None

    if preset:
        key = preset.strip().lower()
        if key not in MODEL_PRESETS:
            raise ValueError(f"Unknown preset {preset!r}. Known: {sorted(MODEL_PRESETS.keys())}")
        meta = MODEL_PRESETS[key]
        return meta["model_id"], meta

    meta = MODEL_PRESETS[default_preset]
    return meta["model_id"], meta
