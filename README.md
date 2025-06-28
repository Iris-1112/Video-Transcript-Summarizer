# Video Transcript Summarizer

## Overview

Video Transcript Summarization is the process of converting spoken content from videos or audio files into concise, readable text that highlights the most important points. This technique enhances accessibility and saves time by enabling users to quickly understand the essence of multimedia content without reviewing the entire recording. It is a significant challenge in the fields of machine learning and natural language processing (NLP).

There are two main approaches to transcript summarization in NLP:

- **Extractive Summarization**
  This approach selects and combines the most important sentences or phrases directly from the original text to form a summary.

- **Abstractive Summarization**
  This approach rewrites and condenses the content in new words, generating a concise summary that may not use the exact sentences from the source.
  
**Chunking Before Summarization**
  Before summarization, lengthy transcripts are divided into smaller sections through a process called chunking. This allows the model to efficiently handle large texts without exceeding input limits. Each chunk is summarized separately, and the results are combined to form the final summary.

In this project, we use chunking followed by abstractive summarization with the BART model from Hugging Face. This approach allows the system to produce clear, human-like summaries from video transcripts.

## Features

- Utilizes OpenAI Whisper to extract transcripts from video/audio files.

- Summarizes large volumes of text using the BART transformer model.

- Supports translation of summaries into multiple Indian regional languages.

- Offers a clean, interactive Streamlit user interface for all processing steps without the need for coding.

- Users can download the generated transcript, summary, and translated files in `.txt` format.

---

![Screenshot 2025-06-28 112511](https://github.com/user-attachments/assets/a17c0f93-47aa-4223-b139-34bd5dde3a02)

---

## Technologies Used

* Streamlit
* OpenAI Whisper
* Hugging Face Transformers (BART)
* Google Translate API (googletrans)
* FFmpeg
* Python


---


## Runtime
Ensure Python 3.10 or 3.11 is installed on your system.


---

## Setup

* **Install dependencies**
  `pip install -r requirements.txt`

* **Install FFmpeg** 

  * Windows: Download from [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/) and add to PATH
  * macOS: `brew install ffmpeg`
  * Linux (Ubuntu/Debian): `sudo apt install ffmpeg`

* **Run the application**
  `streamlit run app.py`

---


