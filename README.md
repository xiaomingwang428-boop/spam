# Email Spam and Phishing Classification Experiments

This repository contains the core experiment implementation for comparing:

- local supervised baselines: TF-IDF, Word2Vec, Linear SVM, Multinomial NB;
- LLM zero-shot classification;
- LLM few-shot prompting without fine-tuning.

The generated outputs, plots, paper drafts, cached API responses, and keys are
intentionally excluded. The raw dataset archive `archive.zip` is included for
reproducible experiments.

## Files

- `dataset_loader.py`: minimal loader for the six CSV files inside `archive.zip`;
- `paper_baseline_classifier.py`: local TF-IDF/Word2Vec baselines;
- `llm_chat_email_classifier.py`: OpenAI-compatible chat-completion experiments;
- `run_gpt_few_shot_sample.ps1`: GPT few-shot sample runner;
- `run_doubao_few_shot_sample.ps1`: Doubao few-shot sample runner;
- `requirements.txt`: Python dependencies;
- `archive.zip`: raw CSV datasets used by the experiments.

## Dataset

The repository includes `archive.zip`. It contains:

```text
CEAS_08.csv
Enron.csv
Ling.csv
Nazario.csv
Nigerian_Fraud.csv
SpamAssasin.csv
```

Labels are interpreted as:

- `0`: legitimate or ham email;
- `1`: spam, phishing, fraud, or malicious email.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Local Baselines

```powershell
python paper_baseline_classifier.py --methods all 
```

```powershell
python paper_baseline_classifier.py --methods all
```

## Outputs

Results are written under `output/`, which is ignored by Git:

- `metrics.json`;
- `predictions.csv`;
- `prediction_cache.sqlite3`;
- trained local `model.joblib` files.
