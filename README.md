# AI Mock Interview Agent

A voice-based, completely local AI mock interview agent designed for AI Engineers. It allows you to practice technical interview questions while simultaneously providing gentle feedback on your English grammar and vocabulary.

## 🚀 Features

- **100% Local Processing** (except TTS): Uses local LLMs and Speech-to-Text to ensure your data stays private and latency is minimal.
- **Voice Interaction**: Talk naturally to the agent, and it will respond with a high-quality voice using `edge-tts`.
- **Dual-Feedback System**: Evaluates both your technical accuracy and your English proficiency.
- **Knowledge Base (RAG)**: Drop your `.txt` or `.md` study notes into the `knowledge_base` folder, and the agent will dynamically quiz you on them.
- **Transcripts**: Automatically saves a markdown transcript of your entire interview (including feedback) into the `transcripts/` folder for later review.

## 🛠️ Tech Stack

- **LLM**: `llama3` via Ollama
- **Embeddings**: `nomic-embed-text` via Ollama
- **Speech-to-Text (STT)**: `faster-whisper` (running locally on CPU/GPU)
- **Text-to-Speech (TTS)**: `edge-tts`
- **RAG / Vector DB**: `ChromaDB` & `LangChain`

## ⚙️ Setup & Installation

### 1. Prerequisites
You must have [Ollama](https://ollama.com/) installed. Pull the required models:
```bash
ollama pull llama3:latest
ollama pull nomic-embed-text
```

### 2. Install Python Dependencies
It is highly recommended to use a virtual environment:
```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

*(Note for Linux users: `pyaudio` requires the system package `portaudio19-dev` to be installed via `sudo apt-get install portaudio19-dev python3-dev` before running `pip install`).*

## 🎙️ Usage

1. Start the interview loop:
   ```bash
   python main.py
   ```
2. Wait for the agent to greet you.
3. Speak into your microphone when you see `[Listening... Speak now!]`.
4. Say **"quit"**, **"stop"**, or **"goodbye"** to end the interview and save your transcript.

## 📚 Customizing the Knowledge Base

1. Place your own `.txt` or `.md` files into the `knowledge_base/` directory. 
2. The next time you run `main.py`, the agent will automatically parse, embed, and store your documents to use as context for your interview questions!
