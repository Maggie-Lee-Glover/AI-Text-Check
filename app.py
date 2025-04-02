import streamlit as st
import re
import textstat
import difflib
import requests
from typing import List

# ------------------------------
# CONFIG
# ------------------------------
st.set_page_config(page_title="AI Text Checker", layout="wide")
st.title("🧠 AI Text Detection & Style Comparison Tool")

# ------------------------------
# FUNCTIONS
# ------------------------------
def split_sentences(text):
    return re.split(r'(?<=[.!?]) +', text.strip())

def calculate_humanness(sentence: str):
    try:
        readability = textstat.flesch_reading_ease(sentence)
        complexity = textstat.sentence_complexity(sentence)
        syllables = textstat.syllable_count(sentence)
        ai_score = 0
        if readability > 60:
            ai_score += 0.2
        if complexity < 1.5:
            ai_score += 0.4
        if syllables < 20:
            ai_score += 0.4
        return round((1 - ai_score) * 100, 2)
    except:
        return 50

def highlight_sentences(text):
    sentences = split_sentences(text)
    highlights = []
    for s in sentences:
        score = calculate_humanness(s)
        if score < 50:
            highlights.append((s, '🔴'))
        elif score < 75:
            highlights.append((s, '🟡'))
        else:
            highlights.append((s, '🟢'))
    return highlights

def compare_to_sample(user_text, sample_text):
    user_sents = split_sentences(user_text.lower())
    sample_sents = split_sentences(sample_text.lower())
    matcher = difflib.SequenceMatcher(None, ' '.join(user_sents), ' '.join(sample_sents))
    return round(matcher.ratio() * 100, 2)

def detect_with_api(text, api_key):
    try:
        url = "https://api.gptzero.me/v2/predict"  # Replace with your preferred API
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.post(url, json={"document": text}, headers=headers)
        if response.status_code == 200:
            result = response.json()
            return result.get("documents", [{}])[0].get("score", None)
        return None
    except:
        return None

# ------------------------------
# INPUTS
# ------------------------------
input_text = st.text_area("Paste text to analyze", height=300)
style_file = st.file_uploader("Upload your writing sample (optional)", type=["txt"])

st.sidebar.header("🔧 Options")
use_api = st.sidebar.checkbox("Use external AI detection API")
api_key = st.sidebar.text_input("Enter API key", type="password")

# ------------------------------
# ANALYSIS
# ------------------------------
if st.button("Analyze") and input_text:
    with st.spinner("Analyzing text..."):
        sample_text = style_file.read().decode("utf-8") if style_file else None

        # LOCAL HEURISTIC
        highlights = highlight_sentences(input_text)
        avg_human_score = round(sum([calculate_humanness(s[0]) for s in highlights]) / len(highlights), 2)

        # STYLE COMPARISON
        style_match = compare_to_sample(input_text, sample_text) if sample_text else None

        # OPTIONAL API DETECTION
        api_score = detect_with_api(input_text, api_key) if (use_api and api_key) else None

    st.subheader("🔍 Humanness Score")
    st.metric("Estimated Human-Likeness", f"{avg_human_score}%")

    if api_score:
        st.metric("Advanced AI Detection (API)", f"{round(api_score*100)}% AI-likely")

    if style_match is not None:
        st.metric("Style Match to Your Writing", f"{style_match}%")

    st.subheader("✨ Sentence Analysis")
    for sentence, level in highlights:
        color = {"🔴": "#ffdddd", "🟡": "#fff6cc", "🟢": "#ddffdd"}[level]
        st.markdown(f"<div style='background-color:{color};padding:5px;border-radius:5px'>{level} {sentence}</div>", unsafe_allow_html=True)
else:
    st.info("Paste some text and click Analyze to begin.")

st.markdown("---")
st.caption("Built for writers, editors, and curious minds 💡")
