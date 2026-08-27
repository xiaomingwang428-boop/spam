# Email Spam and Phishing Classification Experiments

This repository contains the core experiment implementation for comparing:

- local supervised baselines: TF-IDF, Word2Vec, Linear SVM, Multinomial NB;
- LLM zero-shot classification;
- LLM few-shot prompting without fine-tuning.

The original email datasets, generated outputs, plots, paper drafts, cached API
responses, and keys are intentionally excluded.

## Files

- `dataset_loader.py`: minimal loader for the six CSV files inside `archive.zip`;
- `paper_baseline_classifier.py`: local TF-IDF/Word2Vec baselines;
- `llm_chat_email_classifier.py`: OpenAI-compatible chat-completion experiments;
- `run_gpt_few_shot_sample.ps1`: GPT few-shot sample runner;
- `run_doubao_few_shot_sample.ps1`: Doubao few-shot sample runner;
- `requirements.txt`: Python dependencies.

## Dataset

Place an `archive.zip` file in the repository root. It should contain:

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
python paper_baseline_classifier.py --methods all --limit 1000
```

```powershell
python paper_baseline_classifier.py --methods all
```

## GPT Few-Shot Experiment

```powershell
$env:OPENAI_API_KEY = "your_api_key"
$env:OPENAI_BASE_URL = "https://api.jiekou.ai/openai"
$env:OPENAI_MODEL = "gpt-5.5"
.\run_gpt_few_shot_sample.ps1
```

## Doubao Few-Shot Experiment

```powershell
$env:ARK_API_KEY = "your_ark_key"
$env:ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
$env:DOUBAO_MODEL = "doubao-seed-2-1-pro-260628"
.\run_doubao_few_shot_sample.ps1
```

If your Volcengine Ark account uses an endpoint ID, set `DOUBAO_MODEL` to the
`ep-...` value from the console.

## Outputs

Results are written under `output/`, which is ignored by Git:

- `metrics.json`;
- `predictions.csv`;
- `prediction_cache.sqlite3`;
- trained local `model.joblib` files.
