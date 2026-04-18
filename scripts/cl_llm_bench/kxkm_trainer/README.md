# kxkm_trainer — remote LoRA trainer for the CL LLM benchmark

## Purpose

Standalone LoRA fine-tuning script invoked by `LoRATrainerReal`
(`scripts/cl_llm_bench/lora_trainer.py`) on **kxkm-ai** (RTX 4090 24 GB).
One entry point — `train_cl_task.py` — consumes a JSONL dataset and a
base model ID, emits a LoRA adapter plus a parseable manifest.

## Why it lives outside the main package

`scripts/cl_llm_bench/` runs inside this repo's `uv`-managed venv
(NumPy + MLX + POT, Python 3.14). kxkm-ai has its own CUDA torch stack
and no uv pre-installed. We resolve the split two ways:

- The trainer declares its heavy deps (`torch`, `transformers`, `peft`,
  `datasets`, `accelerate`) via **PEP 723 inline metadata** at the top of
  the file, so `uv run train_cl_task.py ...` builds an ephemeral venv on
  kxkm-ai on demand.
- It is **never** imported from the rest of the repo — not by CI, not by
  pytest. The companion test (`test_kxkm_trainer.py`) only does static
  checks (file exists, expected flags present, PEP 723 block present).

## Sync to kxkm-ai

```bash
rsync -av scripts/cl_llm_bench/kxkm_trainer/ kxkm-ai:~/kiki-flow-research-kxkm/
```

The trailing slash matters: contents of `kxkm_trainer/` land directly
under `~/kiki-flow-research-kxkm/` on the remote host.

## Invoke (manually, or via `LoRATrainerReal.build_command`)

```bash
ssh kxkm-ai 'uv run ~/kiki-flow-research-kxkm/train_cl_task.py \
    --base-model Qwen/Qwen3-4B \
    --lora-rank 8 --lora-alpha 16 \
    --learning-rate 1e-4 \
    --n-steps 100 --batch-size 4 \
    --dataset /path/to/data.jsonl \
    --output-dir /path/to/out \
    --seed 0'
```

`LoRATrainerReal.REMOTE_SCRIPT = "~/kiki-flow-research-kxkm/train_cl_task.py"`
and `build_command()` prepends `uv run` so this invocation matches what
the caller emits.

## Dataset format

One JSON object per line:

```json
{"text": "...", "label": 0}
{"text": "...", "label": 1}
```

- `text`: the example sentence/paragraph (string, truncated to 128 tokens)
- `label`: integer class, 0 or 1 (binary classification only — GLUE SST-2,
  CoLA, RTE, BoolQ cast to binary)

The script does an 80/20 train/eval split on the loaded file in memory —
no separate eval file needed.

## Manifest schema

On success, `manifest.json` (and stdout) contains:

```json
{
  "status": "ok",
  "mode": "real",
  "base_model": "Qwen/Qwen3-4B",
  "n_steps": 100,
  "n_samples": 1234,
  "seed": 0,
  "eval_accuracy": 0.78,
  "lora_rank": 8,
  "lora_alpha": 16,
  "timestamp": 1729000000.0
}
```

## LoRA target modules

Default: `["q_proj", "v_proj"]` (Qwen, LLaMA, Mistral family). Falls back
to `["query", "value"]` (BERT/RoBERTa family) on `ValueError`. Anything
else raises — set a compatible `--base-model` or extend `wrap_with_lora`
in the script.
