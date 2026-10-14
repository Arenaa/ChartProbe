# ChartProbe

Code and data for [ChartProbe](https://arxiv.org/abs/2608.13766): a diagnostic study of chart understanding in vision-language models.

When a model misses a chart question, ChartProbe asks which skill failed. Each sample has nine probes with exact, computed answers: three perception, three grounding, and three simple reasoning. Complex-reasoning (CR) sets add one free-form question per sample.

- Preprint: https://arxiv.org/abs/2608.13766
- Project page: https://arenaa.github.io/ChartProbe-webpage/

This repo ships `src/` and packed sets under `data/`. Predictions and checkpoints (`bench_runs/`, `checkpoints/`) are gitignored.

---

## Setup

Chart PNGs under `data/` are Git LFS objects.

```bash
git lfs install
git clone https://github.com/Arenaa/ChartProbe.git
cd ChartProbe
git lfs pull

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU recommended. Gated Hub models need `HF_TOKEN` or `--hub-token`. List presets with `python src/run_benchmark.py --list-presets`. Smoke-test with `--max-samples 8`.

---

## Data

Inventory: [`src/dataset_manifest.json`](src/dataset_manifest.json).

| Path | Role | Samples | Questions / sample |
|------|------|--------:|-------------------:|
| `data/Bar_charts_1000/` | train (ChartProbe-9) | 1000 | 9 |
| `data/CR_bar_500/` | test, ChartNet bars | 500 | 1 |
| `data/CR_pie_500/` | test, ChartNet pies | 500 | 1 |
| `data/CR_CLEVR_500/` | test, CLEVR | 500 | 1 |
| `data/CR_ChartQA_500/` | optional ChartQA (bar + pie) | 500+500 | 1 |

The main paper setup trains on `Bar_charts_1000` and evaluates CR on bar, pie, and CLEVR. ChartQA is held-out.

`run_benchmark.py` takes a bundle directory that contains `samples.jsonl` (and images). For the train pack that is `data/Bar_charts_1000/eval_bundle` (9,000 rows). CR folders use `samples.jsonl` at the folder root.

---

## Evaluation

```bash
python src/run_benchmark.py --preset internvl35-8b \
  --test-bundle data/CR_bar_500 \
  --output-dir bench_runs/CR_bar_500/internvl35-8b
```

Same command for `data/CR_pie_500`, `data/CR_CLEVR_500`, `data/CR_ChartQA_500/bar_charts_500`, `data/CR_ChartQA_500/pie_charts_500`, and `data/Bar_charts_1000/eval_bundle`.

Pass several `--preset` flags, a Hub `--model-id`, or `--lora-adapter` after fine-tuning. Default prompts are CoT (`<think>` / `<answer>`); `--prompt-mode direct` skips CoT.

| Task | Metric |
|------|--------|
| ChartProbe-9 | `avg_type_aware_correct_percent` |
| Complex reasoning | `avg_fuzzy_accuracy_percent` (RapidFuzz) |

```bash
python src/run_benchmark.py --help
```

---

## Fine-tuning

Paper recipe: attention + MLP LoRA, vision frozen, r=32, α=64, 1 epoch, 900/100 chart split (seed 42), train seeds 42/43/44.

Build stage SFT JSONL from the train pack (perception, grounding, reasoning, perception_grounding, all):

```bash
python src/build_stage_sft_splits.py --dataset-dir data/Bar_charts_1000
# writes data_other/chart_qa_sft/stage_experiments/<setup>/{train,val}.jsonl
```

```bash
python src/finetune_lora.py \
  --model internvl35-8b \
  --train-jsonl data_other/chart_qa_sft/stage_experiments/all/train.jsonl \
  --val-jsonl data_other/chart_qa_sft/stage_experiments/all/val.jsonl \
  --output-dir checkpoints/stage/all/internvl35-8b/seed42 \
  --seed 42 --epochs 1.0 --lora-r 32 --lora-alpha 64 \
  --batch-size 1 --grad-accum 8 --learning-rate 1e-4
```

Learning rates: `internvl35-8b` and `llava-v16-mistral-7b` at `1e-4` (batch 1 × accum 8); `qwen35-9b` at `2e-4` (1 × 16).

Probe templates live in [`src/question_templates.json`](src/question_templates.json). Packed `data/` is enough to eval and train; rebuild scripts under `src/build_*.py` and `src/generate_probe_questions.py` are optional.

---

## Citation

```bibtex
@misc{khoshnoodi2026chartprobediagnosticstudyvisual,
  title         = {ChartProbe: A Diagnostic Study on Visual Reasoning through Perception, Grounding, and Simple Reasoning},
  author        = {Mahsa Khoshnoodi and Sarah Adel Bargal},
  year          = {2026},
  eprint        = {2608.13766},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2608.13766}
}
```

## License

MIT. See [`LICENSE`](LICENSE).
