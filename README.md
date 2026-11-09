# 🎥 Semantic YouTube Video Search Engine with Ollama & LLMs

An intelligent semantic video search system designed to search through YouTube video content using natural language prompts. This project automatically retrieves transcripts from YouTube playlists, performs chunking, generates concise summaries using a local LLM (**Qwen 2.5:3b**), computes vector embeddings via **BGE-M3**, and locates the exact video timestamps using Cosine Similarity.

## 📌 Key Features

* 📥 **Automatic Data Extraction:** Fetches metadata and subtitles (English/Vietnamese) from YouTube playlists using `yt-dlp` and `youtube-transcript-api`.

* ✂️ **Smart Chunking Strategy:** Splits video transcripts into 3-minute intervals (180 seconds) with a 20-word overlap to preserve context between chunks.

* 📝 **Automated Text Summarization:** Leverages the local LLM **Qwen 2.5:3b** via Ollama to summarize each transcript chunk into 60 words or fewer.

* 🧠 **Multilingual Vector Embeddings:** Computes 1024-dimensional semantic embedding vectors using **BGE-M3**.

* 🎯 **Precise Timestamp Search:** Enables natural language querying to return direct YouTube links starting at the exact timestamp where the query topic is discussed.

## 🛠️ Tech Stack

* **Language:** Python 3.10+

* **Libraries & Tools:** `pandas`, `numpy`, `scikit-learn`, `ollama`, `yt-dlp`, `youtube-transcript-api`

* **AI Models (Served via Ollama):**

  * **Summarization Model:** `qwen2.5:3b`

  * **Embedding Model:** `bge-m3` (Output dimension: 1024)

## ⚙️ Processing Pipeline

```
[ YouTube Playlist ] 
       │
       ▼ (yt-dlp & transcript API)
[ Raw Transcripts + Timestamps ]
       │
       ▼ (3-min Chunking + 20-word Overlap)
[ Text Chunks ]
       │
       ▼ (Ollama: Qwen 2.5:3b)
[ Concise Summaries ]
       │
       ▼ (Ollama: BGE-M3)
[ Vector Embeddings (1024d) ]
       │
       ▼ (Cosine Similarity Search)
[ Search Results + Direct Timestamp Links ]

```

## 🚀 Installation & Setup (Google Colab / Local Linux)

### 1. Install System & Python Dependencies

```
# Install zstd compression tool
apt-get update -qq && apt-get install -y zstd

# Install required Python packages
pip install pandas numpy scikit-learn yt_dlp youtube_transcript_api ollama

```

### 2. Install & Start Ollama Service

```
# Download and install Ollama
!curl -fsSL https://ollama.com/install.sh | sh

# Launch Ollama server in the background
import subprocess
ollama_process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

```

### 3. Pull Required AI Models

```
# Pull Qwen 2.5 3B for summarization
!ollama pull qwen2.5:3b

# Pull BGE-M3 for embedding generation
!ollama pull bge-m3

```

## 💻 Usage Instructions

### Step 1: Extract Transcripts & Build Dataset

```
from main_script import build_playlist_dataset  # Or execute code directly in notebook

playlist_url = "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID"
df_playlist = build_playlist_dataset(playlist_url, max_duration_hours=2)

# Save raw chunk dataset
df_playlist.to_json("playlist_dataset.json", orient="records", indent=4, force_ascii=False)

```

### Step 2: Summarize Chunks & Generate Embeddings

```
import pandas as pd
import ollama

# 1. Summarize using Qwen 2.5:3b
def summary(text):
    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {"role": "system", "content": "You are a text summarization assistant. Summarize the text under 60 words."},
            {"role": "user", "content": f"Text to summarize:\n{text}"}
        ],
        options={"temperature": 0}
    )
    return response['message']['content'].strip()

df['summary'] = df['text_chunk'].apply(summary)

# 2. Batch Embedding Generation using BGE-M3
response = ollama.embed(model='bge-m3', input=df['summary'].tolist())
df['embedding_summary'] = response.embeddings

# Save dataset with embeddings
df.to_json('youtube_dataset_embedding.json', orient='records', indent=4, force_ascii=False)

```

### Step 3: Run Semantic Search

```
# Execute search query
results = get_video(
    dataset=df,
    query="What is linear regression?",
    rows=3,
    SIMILARITIES_RESULTS_THRESHOLD=0.3
)

# Display matching results
display_results(results, query="What is linear regression?")

```

**Sample Output:**

```
Videos similar to 'What is linear regression?':

- Summary of all Machine Learning algorithms (New version 2025)
  Summary: Linear Regression, or linear regression, is one of the fundamental algorithms...
  YouTube: https://youtu.be/59XV4o8hOBk?t=0
  Similarity: 0.7347
--------------------------------------------------
- All Machine Learning Algorithms in 23 Minutes
  Summary: Linear regression is a basic machine learning algorithm used to predict...
  YouTube: https://youtu.be/ULE78ME1ckQ?t=180
  Similarity: 0.7198
--------------------------------------------------

```

## 📊 JSON Dataset Schema

The processed output dataset `youtube_dataset_embedding.json` contains the following fields:

| Field | Type | Description | 
| ----- | ----- | ----- | 
| `videoId` | `string` | YouTube Video ID | 
| `title` | `string` | Video Title | 
| `start` | `string` | Start timestamp formatted as `HH:MM:SS` | 
| `seconds` | `int` | Start timestamp in seconds (used to generate `?t=seconds` links) | 
| `summary` | `string` | Concise 3-minute chunk summary produced by LLM | 
| `embedding_summary` | `list[float]` | 1024-dimensional vector generated by `bge-m3` | 

## 📝 License

This repository is built and maintained for educational and research purposes.