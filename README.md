# qd2121_dl_midterm

**Team:** Spline Reticulator

**Author:** Thanh Do (qd2121) - only team member

## Summary

Repository for the Deep Learning midterm (ECE-GY 7123 / CS-GY 6953, Fall '25). This repo contains the materials for a Kaggle Math Question Answer Verification competition entry. The objective was to build a verifier model that predicts whether a proposed solution to a math question is correct (True/False). The final approach used Supervised Fine-Tuning (SFT) of a Llama-3.1 8B model with LoRA adapters and prompt engineering.

## Key results

- Final validation accuracy (best model): **89.2%**
- Kaggle private score: **0.86120** — team placed **12th out of 71** teams

## Repository contents

- `report/` — LaTeX source of the project report. Main file: `report/acl_latex.tex`.
- `repro_notebook.ipynb` — reproducibility / inference notebook (used for reproducing experiments and evaluation).
- `training_notebook.ipynb` — training notebook used for iterative fine-tuning experiments.
- `baseline_establisher.ipynb` — training notebook used to establish the baseline validation accuracy value.

## High-level method

- **Dataset:** `ad6398/nyu-dl-teach-maths-comp` (Hugging Face). Each row contains `question`, `answer`, `solution`, and the target `is_correct` boolean.
- **Base model:** `unsloth/Meta-Llama-3.1-8B` (SFT with LoRA adapters). QLoRA was used during initial prototyping for memory efficiency, but was dropped in later runs to utilize the runtime better.
- **Training:** LoRA rank up to `r=32`, `lora_alpha=64`, trained with `SFTTrainer` (TRL) using `adamw_torch`, sequence length 2048, and up to 8,500 training steps in the final runs.
- **Prompting:** a verifier-style prompt that instructs the model to read the solution and output `True`/`False`.

## How to run the notebooks (local / GPU)

### Requirements

- Python 3.10+ (recommended)
- CUDA-capable GPU (A100/40GB recommended for full model experiments) or use Colab with GPU
- The notebooks include dependency/setup cells (install Hugging Face libraries, TRL, tokenizers, etc.). Run those first.

### Basic steps

1. Open `training_notebook.ipynb` and run the dependency/setup cell.
2. Configure any required Hugging Face tokens or dataset paths in the notebook (if prompted).
3. Run training cells — note these experiments are compute-heavy (the team used an A100 with 40GB VRAM and ~17 hours of compute credits for all runs, and ~14 hours for the final model).
4. Use `repro_notebook.ipynb` to run evaluation and reproduce inference results on held-out data.

## CI: building the PDF report

A GitHub Actions workflow is provided at `.github/workflows/build-report.yml`. You can get the latest report by fetching the latest `report-pdf` artifact from the run page.

I also included [a prebuilt report file in the repo](DL_Midterm.pdf) which was submitted to Gradescope. Links to weight files can be found there.

Please note that in order to download the weights file and view the Colab notebooks, you'll need to log in to Google with your NYU account.