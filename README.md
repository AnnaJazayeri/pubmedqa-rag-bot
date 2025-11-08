# PubMedQA LLM Starter

This starter kit helps you build **Medical Question Answering** using **PubMedQA** and **BioGPT / PubMedBERT**.

## What I built
1) **Baseline** (fast): TF–IDF + Logistic Regression (predicts _yes/no/maybe_).  
2) **LLM Pass** (domain model): BioGPT generation on top‐k retrieved context.  
3) **Demo App**: Gradio web UI you can share from Colab.

## Quick start (Google Colab)
1. Open `notebooks/01_load_data_and_baseline.ipynb` in Colab → Run all.  
2. Open `notebooks/02_biogpt_prompting.ipynb` → Run all (uses `microsoft/biogpt`).  
3. Open `notebooks/03_gradio_app.ipynb` → Run all → Get a shareable link for the QA bot.

## Dataset
- Hugging Face: `pubmed_qa` (config: `pqa_labeled`)
- Automatically downloaded via `datasets` library.

## Environment (pip)
See `requirements.txt` for exact versions.

## Project structure
```
pubmedqa_llm_starter/
├─ notebooks/
│  ├─ 01_load_data_and_baseline.ipynb
│  ├─ 02_biogpt_prompting.ipynb
│  └─ 03_gradio_app.ipynb
├─ src/
│  ├─ retrieval.py
│  └─ utils.py
├─ app/
│  └─ templates.txt
├─ data/
├─ requirements.txt
└─ README.md
```

---

## Evaluation ideas
- Baseline accuracy (majority class vs TF–IDF LR) on validation set.
- LLM prompting accuracy on a small subset (e.g., 200 examples).
- Error analysis (where LLM says "Yes" but gold is "No", etc.).
- Ablations: with/without retrieval; different prompt templates.

