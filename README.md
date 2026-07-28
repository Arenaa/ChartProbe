# ChartProbe

Skill-targeted diagnostics for chart-understanding VLMs.

ChartProbe pinpoints where a vision-language model fails on charts, not just whether it does. Every sample is decomposed into 9 fixed probes: 3 Perception, 3 Grounding, and 3 Reasoning, so errors can be traced to a specific stage rather than a single aggregate score. Complex-Reasoning (CR) sets add 1 free-form chain-of-thought question per sample to test end-to-end reasoning.

This release ships code under `src/` and packed eval/train data under `data/`.  
Run outputs go to `bench_runs/` / `checkpoints/` (gitignored).

---

## Contents

1. [Setup](#1-setup)
2. [Repository layout](#2-repository-layout)
3. [Shipped datasets](#3-shipped-datasets)
4. [Run evaluation](#4-run-evaluation)
5. [LoRA fine-tuning](#5-lora-fine-tuning)
6. [Question templates](#6-question-templates)
7. [Rebuild data (optional)](#7-rebuild-data-optional)
8. [Tips](#8-tips)
9. [Script index](#9-script-index)

---

## 1. Setup

```bash
# Git LFS required (chart PNGs under data/ are LFS objects)
git lfs install

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export HF_TOKEN=...   # gated Hub models (optional)
# or: export HUGGING_FACE_HUB_TOKEN=...
```

After clone: `git lfs pull` if images are still pointer files.

GPU recommended. List model presets:

```bash
python src/run_benchmark.py --list-presets
```

---

## 2. Repository layout

```text
ChartProbe/
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/                         # packed train + eval sets (see §3)
└── src/
    ├── run_benchmark.py          # CoT eval + scoring
    ├── finetune_lora.py          # LoRA trainer
    ├── question_templates.json   # ChartProbe-9 template source of truth
    ├── templates.py              # load / format templates
    ├── dataset_manifest.json     # path inventory
    └── …
```

| Path | Purpose |
|------|---------|
| `data/` | Train bars + CR / ChartQA eval packs |
| `data_other/chart_qa_sft/` | Stage SFT JSONL (generated locally) |
| `checkpoints/` | LoRA adapters |
| `bench_runs/` | Predictions + metrics |

---

## 3. Shipped datasets

Canonical inventory: [`src/dataset_manifest.json`](src/dataset_manifest.json).

| Folder | Role | Samples | Q / sample | Notes |
|--------|------|--------:|-----------:|-------|
| `data/Bar_charts_1000/` | train | 1000 | 9 | ChartNet bars + ChartProbe-9 |
| `data/CR_bar_500/` | test (CR) | 500 | 1 | ChartNet bar complex reasoning |
| `data/CR_pie_500/` | test (CR) | 500 | 1 | ChartNet pie complex reasoning |
| `data/CR_CLEVR_500/` | test (CR) | 500 | 1 | CLEVR complex reasoning |
| `data/CR_ChartQA_500/` | test (ChartQA) | 500+500 | 1 | ChartQA bar + pie (not CoT CR) |

**Main experiment:** train on `Bar_charts_1000`, evaluate CR on bar / pie / CLEVR.  
**ChartQA** is an optional eval set (500 bar + 500 pie).

### What is an eval bundle?

An **eval bundle** is the ready-to-run question list for `run_benchmark.py`:

```text
…/eval_bundle/
  samples.jsonl    # one row per question (image, Q, gold, stage, …)
  images/          # optional local image copies / links
```

For `Bar_charts_1000`, `eval_bundle/samples.jsonl` has **9000** rows (9 × 1000).  
CR folders put `samples.jsonl` (+ `images/`) at the folder root.

### `Bar_charts_1000` internals

| Path | Purpose |
|------|---------|
| `images/` | 1000 chart PNGs |
| `eval_bundle/` | ChartProbe-9 questions used for train/eval |
| `samples.jsonl` / `metadata.json` | One row per chart |
| `annotations/` | Per-chart bbox JSON (grounding rebuild) |
| `dataset/questions.json` | Intermediate question dump |

### Eval entrypoints

```text
data/Bar_charts_1000/eval_bundle
data/CR_bar_500
data/CR_pie_500
data/CR_CLEVR_500
data/CR_ChartQA_500/bar_charts_500
data/CR_ChartQA_500/pie_charts_500
```

Upstream sources: [ChartNet](https://huggingface.co/datasets/ibm-granite/ChartNet), CLEVR val, ChartQA.

---

## 4. Run evaluation

Entrypoint: `src/run_benchmark.py`.

### Complex reasoning (fuzzy scoring)

```bash
python src/run_benchmark.py --preset internvl35-8b \
  --test-bundle data/CR_bar_500 \
  --output-dir bench_runs/CR_bar_500/internvl35-8b

python src/run_benchmark.py --preset internvl35-8b \
  --test-bundle data/CR_pie_500 \
  --output-dir bench_runs/CR_pie_500/internvl35-8b

python src/run_benchmark.py --preset internvl35-8b \
  --test-bundle data/CR_CLEVR_500 \
  --output-dir bench_runs/CR_CLEVR_500/internvl35-8b
```

### ChartQA (optional)

```bash
python src/run_benchmark.py --preset internvl35-8b \
  --test-bundle data/CR_ChartQA_500/bar_charts_500 \
  --output-dir bench_runs/ChartQA_bar_500/internvl35-8b

python src/run_benchmark.py --preset internvl35-8b \
  --test-bundle data/CR_ChartQA_500/pie_charts_500 \
  --output-dir bench_runs/ChartQA_pie_500/internvl35-8b
```

### ChartProbe-9 on the train pack (type-aware scoring)

```bash
python src/run_benchmark.py --preset internvl35-8b \
  --test-bundle data/Bar_charts_1000/eval_bundle \
  --output-dir bench_runs/Bar_charts_1000/internvl35-8b
```

### Multiple models / Hub ids / LoRA

```bash
python src/run_benchmark.py \
  --preset internvl35-8b --preset qwen35-9b --preset llava-v16-mistral-7b \
  --test-bundle data/CR_bar_500 \
  --output-dir bench_runs/CR_bar_500/sweep

python src/run_benchmark.py \
  --model-id OpenGVLab/InternVL3_5-8B-HF \
  --test-bundle data/CR_bar_500

python src/run_benchmark.py --preset internvl35-8b \
  --lora-adapter checkpoints/stage/all/internvl35-8b/seed42 \
  --test-bundle data/CR_bar_500 \
  --output-dir bench_runs/cr_bar_ft_all_seed42
```

### Metrics

| Task | Primary metric |
|------|----------------|
| ChartProbe-9 | `avg_type_aware_correct_percent` |
| Complex reasoning | `avg_fuzzy_accuracy_percent` (RapidFuzz) |

Default prompt mode is **CoT** (`<think>` / `<answer>`). Use `--prompt-mode direct` for non-CoT baselines.

```bash
python src/run_benchmark.py --help
# --max-samples N  --prompt-mode cot|direct  --hub-token TOKEN  --cuda-device 0
```

---

## 5. LoRA fine-tuning

Paper-style recipe (attention + MLP LoRA, vision frozen): **r=32**, **α=64**, **1 epoch**, chart split **900/100** (seed 42), train seeds **42/43/44**.

### 5.1 Build stage SFT JSONL from `Bar_charts_1000`

```bash
# Five setups: perception | grounding | reasoning | perception_grounding | all
python src/build_stage_sft_splits.py \
  --dataset-dir data/Bar_charts_1000
# writes under data_other/chart_qa_sft/stage_experiments/<setup>/{train,val}.jsonl
```

Controls (optional):

```bash
python src/build_stage_sft_volume_matched.py --help
python src/build_stage_sft_volume_matched_all.py --help
python src/build_stage_sft_scrambled.py --help
```

### 5.2 Train

```bash
python src/finetune_lora.py \
  --model internvl35-8b \
  --train-jsonl data_other/chart_qa_sft/stage_experiments/all/train.jsonl \
  --val-jsonl data_other/chart_qa_sft/stage_experiments/all/val.jsonl \
  --output-dir checkpoints/stage/all/internvl35-8b/seed42 \
  --seed 42 \
  --epochs 1.0 \
  --lora-r 32 \
  --lora-alpha 64 \
  --batch-size 1 \
  --grad-accum 8 \
  --learning-rate 1e-4
```

| Preset | LR | batch × grad_accum |
|--------|----|--------------------|
| `internvl35-8b` | `1e-4` | `1 × 8` |
| `llava-v16-mistral-7b` | `1e-4` | `1 × 8` |
| `qwen35-9b` | `2e-4` | `1 × 16` |

Evaluate with `--lora-adapter` as in [§4](#4-run-evaluation).

---

## 6. Question templates

**Source of truth:** [`src/question_templates.json`](src/question_templates.json)  
Loaded via [`src/templates.py`](src/templates.py) (families: `bar`, `pie`, `clevr`).

```bash
python src/generate_probe_questions.py --dataset Bar_charts_1000 \
  --templates src/question_templates.json

python src/build_question_templates_catalog.py
# refreshes src/question_templates_catalog.json
```

| Stage | Bar / pie | CLEVR |
|-------|-----------|-------|
| Perception | count, height, trend | count_total, count_size, count_shape |
| Grounding | bbox, value→label, value lookup | bbox, attribute, spatial |
| Reasoning | rank, arithmetic, multi-step | compare, open_count, relation |

---

## 7. Rebuild data (optional)

Packed `data/` is enough to eval and finetune. Rebuild only if you change templates or regenerate splits:

```bash
python src/generate_probe_questions.py --help
python src/build_complex_reasoning_sets.py --help
python src/build_clevr_complex_reasoning.py --help
python src/build_chartqa_bar_pie_500.py --help
python src/extend_probes_to_500.py --help
```

Probe-500 ChartProbe-9 folders (`bar_charts_500`, `pie_charts_500`, `clevr_500`) are **not** shipped here; build them locally if needed.

---

## 8. Tips

- Gated models: `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` or `--hub-token`.
- Smoke test: `--max-samples 8` before full runs.
- Do not commit `bench_runs/`, `checkpoints/`, or `results/`. `data/` is large; prefer Git LFS or a release tarball if publishing.

---

## 9. Script index

| Script | Purpose |
|--------|---------|
| `src/run_benchmark.py` | CoT eval + scoring |
| `src/finetune_lora.py` | LoRA fine-tune |
| `src/generate_probe_questions.py` | Emit ChartProbe-9 questions |
| `src/build_stage_sft_splits.py` | Stage SFT splits from bar-1000 |
| `src/build_complex_reasoning_sets.py` | ChartNet CR bar/pie |
| `src/build_clevr_complex_reasoning.py` | CLEVR CR |
| `src/build_chartqa_bar_pie_500.py` | ChartQA bar/pie pack |
| `src/extend_probes_to_500.py` | Grow 100 → 500 probes / CR |
| `src/question_templates.json` | Editable ChartProbe-9 templates |
| `src/templates.py` | Load / format templates |
| `src/model_presets.py` | Named Hub checkpoints |
| `src/eval_metrics.py` | Type-aware vs fuzzy metrics |
| `src/test_bundle.py` | Load `samples.jsonl` + images |

```bash
python src/<script>.py --help
```

---

## License

See [`LICENSE`](LICENSE).
