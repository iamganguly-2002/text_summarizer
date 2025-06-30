import streamlit as st
from preprocess import clean_text
from utils import create_tokenizer, encode_and_pad
from model import build_model
import numpy as np
import keras

# Dummy summarization logic for now (replace with real one if trained)
def dummy_summarize(text):
    sentences = text.split('. ')
    return '. '.join(sentences[:2]) + '.'

st.set_page_config(page_title="Text Summarizer", layout="wide")

st.title("📝 Text Summarizer - Seq2Seq LSTM")

input_text = st.text_area("Enter a paragraph to summarize", height=300)

if st.button("Summarize"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_text = clean_text(input_text)
        summary = dummy_summarize(input_text)  # Replace this with real model prediction
        st.subheader("📌 Summary:")
        st.success(summary)
