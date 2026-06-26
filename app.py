"""
Gradio demo for the fine-tuned DistilBERT emotion classifier.

Set MODEL_ID to your Hugging Face Hub repo (e.g. "your-username/distilbert-emotion")
once you've pushed the trained model, or point it at a local output_dir from train.py.
"""

import os

import gradio as gr
from transformers import pipeline

MODEL_ID = os.environ.get("MODEL_ID", "distilbert-emotion")

classifier = pipeline("text-classification", model=MODEL_ID, top_k=None)


def predict(text):
    if not text or not text.strip():
        return {}
    results = classifier(text)[0]
    return {item["label"]: item["score"] for item in results}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, placeholder="Type a sentence...", label="Text"),
    outputs=gr.Label(num_top_classes=6, label="Predicted Emotion"),
    title="Emotion Classifier — Fine-Tuned DistilBERT",
    description=(
        "DistilBERT fine-tuned on the dair-ai/emotion dataset to classify text into "
        "six emotions: sadness, joy, love, anger, fear, surprise."
    ),
    examples=[
        "I can't believe I won the lottery, this is amazing!",
        "I miss my family so much it hurts.",
        "How dare you talk to me like that!",
    ],
)

if __name__ == "__main__":
    demo.launch()
