# Emotion Classifier — Fine-Tuned DistilBERT

Fine-tunes `distilbert-base-uncased` on the [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset
to classify text into six emotions: **sadness, joy, love, anger, fear, surprise**. Trained with the Hugging Face
`Trainer`, published to the Hugging Face Hub, and served via a Gradio demo.

## Setup

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Train

```bash
python train.py --epochs 4 --batch_size 16
```

To train and push the model straight to your Hugging Face Hub account:

```bash
huggingface-cli login
python train.py --push_to_hub --hub_model_id <your-username>/distilbert-emotion
```

This logs F1 (weighted) and accuracy on the validation set each epoch, evaluates on the held-out test set at the end,
and saves the model to `./distilbert-emotion`.

## Run the demo

```bash
set MODEL_ID=<your-username>/distilbert-emotion   # or leave unset to use local ./distilbert-emotion
python app.py
```

Opens a local Gradio UI for real-time inference.

## Project layout

- `train.py` — tokenization, Trainer setup, metrics, training loop, optional Hub push
- `app.py` — Gradio interface for inference
- `requirements.txt` — pinned dependencies
