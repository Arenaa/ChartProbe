#!/usr/bin/env python3
"""
LoRA fine-tune chart QA on HF VLMs (Qwen3.5-9B, InternVL3.5-8B-HF,
LLaVA-1.6 / OneVision, Granite Vision 4.1).

Recommended strategy (single L40S 46GB):
  1. LoRA on language-model attention/MLP only; freeze vision tower.
  2. Match eval prompts (--prompt-mode cot matches run_benchmark).
  3. Train on bar_charts_1000 with chart-level train/val split (export script).
  4. bf16 LoRA first; use --use-4bit if OOM (requires bitsandbytes).

Usage:
  python src/build_sft_jsonl.py --dataset-dir bar_charts_1000
  python src/finetune_lora.py --model qwen35-9b --train-jsonl data_other/chart_qa_sft/bar_charts_1000/train.jsonl
  python src/finetune_lora.py --model granite-4.1-3b --train-jsonl data_other/chart_qa_sft/stage_experiments/all/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

from run_benchmark import build_cot_prompt, build_direct_prompt
from vlm_processing import patch_granite_create_causal_mask, prepare_vlm_inference
from model_presets import resolve_model_id

logger = logging.getLogger(__name__)

MODEL_ALIASES = {
    "qwen35-9b": "qwen35-9b",
    "internvl35-8b": "internvl35-8b",
    "llava-onevision-7b": "llava-onevision-7b",
    "llava-v16-mistral-7b": "llava-v16-mistral-7b",
    # Chart VLM (Hub: ibm-granite/granite-vision-4.1-4b). granite-4.1-3b is an alias for the same.
    "granite-vision-4.1-4b": "granite-vision-4.1-4b",
    "granite-4.1-3b": "granite-4.1-3b",
    "paligemma-3b-pt-448": "paligemma-3b-pt-448",
}


def _is_paligemma_family(model_id: str) -> bool:
    mid = (model_id or "").lower()
    return "paligemma" in mid or mid.endswith("/chartgemma") or mid == "ahmed-masry/chartgemma"


def _hub_token() -> str | None:
    import os

    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def build_assistant_text(answer: str, *, prompt_mode: str) -> str:
    ans = (answer or "").strip()
    if prompt_mode == "direct":
        return ans
    return (
        "<think>\n\n</think>\n\n"
        f"<answer>\n{ans}\n</answer>"
    )


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def lora_target_module_names(model) -> list[str]:
    """Language-only LoRA targets; skip vision / projector unless user unfreezes."""
    names: set[str] = set()
    for name, module in model.named_modules():
        lname = name.lower()
        if any(x in lname for x in ("vision", "visual", "vit", "mlp1", "img", "merger")):
            continue
        if module.__class__.__name__ == "Linear" and "lm_head" not in lname:
            leaf = name.split(".")[-1]
            if leaf in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}:
                names.add(leaf)
    if not names:
        names = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    return sorted(names)


# LLaVA-OneVision expects a batch dim on vision metadata tensors.
_KEEP_BATCH_DIM_KEYS = frozenset({"image_sizes", "batch_num_images"})


def _ensure_batch_dim(val: torch.Tensor) -> torch.Tensor:
    """OneVision crashes if batch_num_images is a 0-d scalar."""
    if val.dim() == 0:
        return val.view(1)
    return val


@dataclass
class ChartSFTCollator:
    processor: object

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch: dict[str, torch.Tensor] = {}
        keys = features[0].keys()
        for key in keys:
            vals = [f[key] for f in features]
            if key == "labels":
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    vals, batch_first=True, padding_value=-100
                )
            elif key in ("input_ids", "attention_mask"):
                pad_id = (
                    self.processor.tokenizer.pad_token_id
                    if getattr(self.processor, "tokenizer", None) is not None
                    else 0
                )
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    vals, batch_first=True, padding_value=pad_id
                )
            elif key in _KEEP_BATCH_DIM_KEYS:
                v = vals[0] if len(vals) == 1 else torch.cat(vals, dim=0)
                batch[key] = _ensure_batch_dim(v)
            elif key == "pixel_values":
                if len(features) == 1:
                    v = vals[0]
                    # (C, H, W) -> (1, C, H, W) for vision towers that require a batch dim.
                    batch[key] = v.unsqueeze(0) if v.dim() == 3 else v
                elif vals[0].dim() >= 5:
                    batch[key] = torch.cat(vals, dim=0)
                elif vals[0].dim() == 4 and vals[0].shape[1] == 3:
                    batch[key] = torch.cat(vals, dim=0)
                elif vals[0].dim() == 3:
                    batch[key] = torch.stack(vals, dim=0)
                elif vals[0].dim() == 2:
                    batch[key] = torch.cat(vals, dim=0)
                else:
                    batch[key] = torch.stack(vals, dim=0)
            elif key.endswith("_grid_thw"):
                batch[key] = torch.cat(vals, dim=0)
            elif key == "mm_token_type_ids":
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    vals, batch_first=True, padding_value=0
                )
            elif key == "token_type_ids":
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    vals, batch_first=True, padding_value=0
                )
            elif torch.is_tensor(vals[0]):
                batch[key] = vals[0] if len(vals) == 1 else torch.stack(vals, dim=0)
            else:
                batch[key] = vals[0]
        return batch


class ChartQASFTDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        *,
        processor,
        prompt_mode: str = "cot",
        max_length: int = 2048,
        model_id: str = "",
    ):
        self.rows = rows
        self.processor = processor
        self.prompt_mode = prompt_mode
        self.max_length = max_length
        self.model_id = model_id
        self.paligemma = _is_paligemma_family(model_id)

    def __len__(self) -> int:
        return len(self.rows)

    def _encode_row_paligemma(self, row: dict) -> dict[str, torch.Tensor]:
        """PaliGemma / ChartGemma: processor(text=..., images=..., suffix=answer)."""
        image = Image.open(row["image"]).convert("RGB")
        user_text = (
            build_direct_prompt(row["question"])
            if self.prompt_mode == "direct"
            else build_cot_prompt(row["question"])
        )
        assistant_text = build_assistant_text(row["answer"], prompt_mode=self.prompt_mode)
        text = user_text if "<image>" in user_text else f"<image>{user_text}"
        full_inputs = self.processor(
            text=text,
            images=image,
            suffix=assistant_text,
            return_tensors="pt",
        )
        input_ids = full_inputs["input_ids"].squeeze(0)
        if "labels" in full_inputs:
            labels = full_inputs["labels"].squeeze(0)
        else:
            # Fallback: mask everything before the suffix by re-encoding prompt-only.
            prompt_only = self.processor(text=text, images=image, return_tensors="pt")
            prompt_len = int(prompt_only["input_ids"].shape[1])
            labels = input_ids.clone()
            labels[: min(prompt_len, labels.shape[0])] = -100

        if input_ids.shape[0] > self.max_length:
            prompt_len = int((labels == -100).sum().item())
            if prompt_len >= self.max_length:
                raise ValueError(
                    f"Prompt alone exceeds max_length={self.max_length} for chart {row.get('chart_id')}"
                )
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]

        out: dict[str, torch.Tensor] = {"input_ids": input_ids, "labels": labels}
        if "attention_mask" in full_inputs:
            am = full_inputs["attention_mask"].squeeze(0)
            out["attention_mask"] = am[: input_ids.shape[0]]
        for key, val in full_inputs.items():
            if key in out or not torch.is_tensor(val):
                continue
            if key == "pixel_values":
                # Keep (1, C, H, W). Collator batch_size=1 passes tensors through;
                # SigLIP requires 4D pixel_values.
                out[key] = val if val.dim() >= 4 else val.unsqueeze(0)
            else:
                out[key] = val.squeeze(0) if val.dim() > 0 and val.shape[0] == 1 else val
        return out

    def _encode_row(self, row: dict) -> dict[str, torch.Tensor]:
        if self.paligemma:
            return self._encode_row_paligemma(row)

        image = Image.open(row["image"]).convert("RGB")
        user_text = (
            build_direct_prompt(row["question"])
            if self.prompt_mode == "direct"
            else build_cot_prompt(row["question"])
        )
        assistant_text = build_assistant_text(row["answer"], prompt_mode=self.prompt_mode)

        user_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            }
        ]
        full_messages = user_messages + [
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
        ]

        prompt_inputs = self.processor.apply_chat_template(
            user_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        full_inputs = self.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = full_inputs["input_ids"].squeeze(0)
        has_image = "pixel_values" in full_inputs
        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        if input_ids.shape[0] > self.max_length:
            if has_image and prompt_len >= self.max_length:
                raise ValueError(
                    f"Prompt alone exceeds max_length={self.max_length} for chart {row.get('chart_id')}"
                )
            input_ids = input_ids[: self.max_length]

        labels = input_ids.clone()
        prompt_len = min(prompt_len, labels.shape[0])
        labels[:prompt_len] = -100

        out: dict[str, torch.Tensor] = {"input_ids": input_ids, "labels": labels}
        if "attention_mask" in full_inputs:
            am = full_inputs["attention_mask"].squeeze(0)
            out["attention_mask"] = am[: input_ids.shape[0]]
        for key, val in full_inputs.items():
            if key in out or not torch.is_tensor(val):
                continue
            if key.endswith("_grid_thw"):
                out[key] = val
            elif key == "mm_token_type_ids":
                out[key] = val.squeeze(0)[: input_ids.shape[0]]
            elif key in _KEEP_BATCH_DIM_KEYS:
                out[key] = _ensure_batch_dim(val)
            elif key == "pixel_values" and val.dim() >= 5:
                out[key] = val
            else:
                out[key] = val.squeeze(0) if val.dim() > 0 and val.shape[0] == 1 else val
        return out

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        n = len(self.rows)
        last_err: Exception | None = None
        for off in range(n):
            try:
                return self._encode_row(self.rows[(idx + off) % n])
            except ValueError as exc:
                if "exceeds max_length" not in str(exc):
                    raise
                last_err = exc
        raise RuntimeError(
            f"No sample fits max_length={self.max_length} in dataset of {n} rows"
        ) from last_err


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=sorted(MODEL_ALIASES), required=True)
    ap.add_argument("--train-jsonl", type=Path, default=repo / "data_other/chart_qa_sft/bar_charts_1000/train.jsonl")
    ap.add_argument("--val-jsonl", type=Path, default=repo / "data_other/chart_qa_sft/bar_charts_1000/val.jsonl")
    ap.add_argument("--output-dir", type=Path, default=repo / "checkpoints/chart_qa_lora")
    ap.add_argument("--prompt-mode", choices=("cot", "direct"), default="cot")
    ap.add_argument("--torch-dtype", default="bfloat16")
    ap.add_argument("--use-4bit", action="store_true")
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--logging-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--max-val-samples", type=int, default=0)
    ap.add_argument(
        "--max-bar-count",
        type=int,
        default=0,
        help="Drop rows with bar_count above this (0 = no filter; use ~45 for Qwen 4-bit on L40S)",
    )
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    ap.add_argument(
        "--init-lora-adapter",
        type=Path,
        default=None,
        help="Warm-start from an existing LoRA adapter (continue training) instead of random init",
    )
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    preset = MODEL_ALIASES[args.model]
    model_id, _ = resolve_model_id(preset=preset, model_id=None)
    out_dir = args.output_dir / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(args.train_jsonl)
    val_rows = load_jsonl(args.val_jsonl) if args.val_jsonl.is_file() else []
    if args.max_bar_count > 0:
        before = len(train_rows)
        train_rows = [r for r in train_rows if int(r.get("bar_count") or 0) <= args.max_bar_count]
        logger.info(
            "max_bar_count<=%s: kept %d / %d train rows",
            args.max_bar_count,
            len(train_rows),
            before,
        )
        before = len(val_rows)
        val_rows = [r for r in val_rows if int(r.get("bar_count") or 0) <= args.max_bar_count]
        logger.info(
            "max_bar_count<=%s: kept %d / %d val rows",
            args.max_bar_count,
            len(val_rows),
            before,
        )
    if args.max_train_samples > 0:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_rows = val_rows[: args.max_val_samples]

    dtype = getattr(torch, args.torch_dtype)
    if "granite" in model_id.lower():
        patch_granite_create_causal_mask()
        logger.info("Using Granite causal-mask patch for %s", model_id)
    hub_token = _hub_token()
    if _is_paligemma_family(model_id) and "google/paligemma" in model_id.lower() and not hub_token:
        raise SystemExit(
            f"{model_id} is gated. Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN "
            "(and accept the model license on Hugging Face) before training."
        )
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, token=hub_token
    )

    load_kw: dict = dict(
        trust_remote_code=True, dtype=dtype, low_cpu_mem_usage=True, token=hub_token
    )
    if args.use_4bit:
        from transformers import BitsAndBytesConfig

        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kw["device_map"] = "auto"
    model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kw)
    if not args.use_4bit:
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    prepare_vlm_inference(processor, model, model_id=model_id)

    for name, param in model.named_parameters():
        if any(x in name.lower() for x in ("vision", "visual", "vit", "mlp1", "merger")):
            param.requires_grad = False

    from peft import LoraConfig, PeftModel, get_peft_model

    init_adapter = args.init_lora_adapter.resolve() if args.init_lora_adapter else None
    if init_adapter is not None:
        if not init_adapter.is_dir():
            raise SystemExit(f"--init-lora-adapter not found: {init_adapter}")
        if args.use_4bit:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model)
        logger.info("Warm-starting LoRA from %s", init_adapter)
        model = PeftModel.from_pretrained(model, str(init_adapter), is_trainable=True)
        target_modules = lora_target_module_names(model)
    else:
        target_modules = lora_target_module_names(model)
        logger.info("LoRA target modules: %s", target_modules)
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.print_trainable_parameters()

    train_ds = ChartQASFTDataset(
        train_rows,
        processor=processor,
        prompt_mode=args.prompt_mode,
        max_length=args.max_length,
        model_id=model_id,
    )
    eval_ds = (
        ChartQASFTDataset(
            val_rows,
            processor=processor,
            prompt_mode=args.prompt_mode,
            max_length=args.max_length,
            model_id=model_id,
        )
        if val_rows
        else None
    )

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        bf16=args.torch_dtype == "bfloat16",
        fp16=args.torch_dtype == "float16",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=args.eval_steps if eval_ds else None,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=ChartSFTCollator(processor),
    )
    trainer.train()
    trainer.save_model(str(out_dir / "lora_adapter"))
    processor.save_pretrained(str(out_dir / "processor"))
    (out_dir / "train_config.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "preset": preset,
                "prompt_mode": args.prompt_mode,
                "train_jsonl": str(args.train_jsonl),
                "val_jsonl": str(args.val_jsonl),
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "target_modules": target_modules,
                "init_lora_adapter": str(init_adapter) if init_adapter else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Saved LoRA adapter to %s", out_dir / "lora_adapter")


if __name__ == "__main__":
    main()
