import streamlit as st
import whisper
from transformers import pipeline, AutoTokenizer
from googletrans import Translator
import tempfile
import os
import re

languages = {
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "bn": "Bengali",
    "ur": "Urdu",
    "or": "Odia"
}

def transcription(vid_path):
    model = whisper.load_model("base")
    output = model.transcribe(vid_path)
    return output["text"]

def preprocess(text):
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    connectives = [' and ', ' but ', ' so ', ' for example ', ' now ', ' then ', ' thus ', ' however ']
    for connector in connectives:
        text = text.replace(connector, f'.{connector}')
    text = text.replace(' uh ', '. ')
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def chunking(text, tokenizer, max_tokens=950):
    input = tokenizer(text, return_tensors='pt', truncation=False)
    input_ids = input["input_ids"][0]
    input = []
    for i in range(0, len(input_ids), max_tokens):
        chunk_ids = input_ids[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        input.append(chunk_text)
    return input

def summarization(text):
    text = preprocess(text)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    chunks = chunking(text, tokenizer)
    summary = ""
    for chunk in chunks:
        try:
            result = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
            summary += result[0]['summary_text'] + " "
        except Exception:
            continue
    return summary.strip()

def translation(text, lang):
    translator = Translator()
    translation = translator.translate(text, dest=lang)
    return translation.text

st.title("Video Transcript Summarizer")
# Step 1: Transcription
st.header("Step 1: Transcription")
upload_vid = st.file_uploader("Upload a video file", type=["mp4", "wav", "mkv", "mp3"])
if upload_vid is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(upload_vid.name)[1]) as tmpfile:
        tmpfile.write(upload_vid.read())
        tmpfile_path = tmpfile.name
    if st.button("Transcribe Video"):
        with st.spinner("Transcribing..."):
            transcript = transcription(tmpfile_path)
            st.success("Transcription complete!")
            st.download_button("Download Transcript", transcript, file_name="transcript.txt")
    if os.path.exists(tmpfile_path):
        os.remove(tmpfile_path)

# Step 2: Summarization
st.header("Step 2: Summarization")
upload_transcript = st.file_uploader("Upload transcript file (.txt)", type=["txt"], key="transcript")
if upload_transcript is not None:
    transcript_text = upload_transcript.read().decode("utf-8")
    if st.button("Summarize Transcript"):
        with st.spinner("Summarizing..."):
            summary = summarization(transcript_text)
            st.success("Summarization complete!")
            st.download_button("Download Summary", summary, file_name="summary.txt")

# Step 3: Translation
st.header("Step 3: Translation")
upload_summary = st.file_uploader("Upload summary file (.txt)", type=["txt"], key="summary")
lang_code = st.selectbox("Select language", options=list(languages.keys()), format_func=lambda x: f"{x} - {languages[x]}")
if upload_summary is not None and lang_code:
    summary_text = upload_summary.read().decode("utf-8")
    if st.button("Translate Summary"):
        with st.spinner("Translating..."):
            translated_text = translation(summary_text, lang_code)
            st.success("Translation complete!")
            st.download_button("Download Translation", translated_text, file_name=f"translated_{lang_code}.txt")

st.markdown("---")
