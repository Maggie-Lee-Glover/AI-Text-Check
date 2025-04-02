import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import difflib

# NLTK for sentence tokenization
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# Load model (fine-tuned for AI/human text classification)
tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")

# Page layout
st.set_page_config(page_title="AI Text Detector", layout="wide")
st.title("🧠 AI Text Detector - Check for Humanness")

# Input fields
user_input = st.text_area("Paste the text you want to analyze:", height=250)
user_reference = st.text_area("(Optional) Paste your own previous writing for comparison:", height=150)

# Button to trigger detection
if st.button("Analyze Text") and user_input:
    with st.spinner("Analyzing text..."):
        # Split into sentences
        sentences = sent_tokenize(user_input)
        suspicious_sentences = []
        humanness_scores = []

        for sent in sentences:
            inputs = tokenizer(sent, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                human_prob = probs[0][0].item()
                humanness_scores.append(human_prob)

                if human_prob < 0.5:
                    suspicious_sentences.append((sent, human_prob))

        avg_score = sum(humanness_scores) / len(humanness_scores)

        st.subheader("🔎 Results")
        st.markdown(f"**Overall Humanness Score:** `{avg_score * 100:.2f}%`")

        if suspicious_sentences:
            st.warning("Suspicious sentences (likely AI-generated):")
            for sent, score in suspicious_sentences:
                st.markdown(f"> *{sent}* — `{score * 100:.2f}% human-like`")
        else:
            st.success("No suspicious sentences detected. Your writing looks human!")

        # Optional comparison to user's writing
        if user_reference:
            st.subheader("🧬 Similarity Check to Your Writing")
            similarity = difflib.SequenceMatcher(None, user_input, user_reference).ratio()
            st.markdown(f"**Similarity to your own writing:** `{similarity * 100:.2f}%`")
            if similarity < 0.5:
                st.info("This writing differs significantly from your reference style.")
            else:
                st.success("This writing is quite similar to your reference style.")

st.markdown("---")
st.markdown("Made with ❤️ using RoBERTa AI detection and Streamlit.")
