# Gemma-AI-Educational-Suite

Offline AI tutoring suite powered by Gemma 3n. Upload textbooks, ask questions, generate quizzes - all without internet.

## 🎯 Gemma 3n Implementation

This application leverages **Gemma 3n models** through Unsloth for optimized inference:

```python
from unsloth import FastModel
# Core Gemma 3n integration
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3n-E4B-it",  # Supports 2B/4B/5B/8B variants
    dtype=None,
    max_seq_length=1024,
    load_in_4bit=True,  # Memory optimization
    full_finetuning=False,
)
```

**Conversation Format** (Gemma 3n specific):
```python
conversation_text = ""
for msg in messages:
    if msg['role'] == 'user':
        conversation_text += f"<start_of_turn>user\n{msg['content']}<end_of_turn>\n"
    else:
        conversation_text += f"<start_of_turn>model\n{msg['content']}<end_of_turn>\n"
conversation_text += "<start_of_turn>model\n"
```

## 🚀 Key Features

- **Complete Offline Operation**: Download models once, use forever without internet
- **Multimodal Learning**: Text, images, audio (recording + file import), screen capture, documents
- **Educational Tools**: Quiz generation, concept mapping, homework assistance, skill assessment
- **Document Support**: PDF (with page selection), DOCX, TXT, MD, HTML, code files
- **Smart Context Management**: 32K token window with intelligent usage tracking
- **Model Flexibility**: Supports all Gemma 3n variants (2B to 8B parameters)

## 🛠️ Setup

```bash
# Clone repository
git clone https://github.com/DaltonPayne/Gemma-AI-Educational-Suite.git
cd Gemma-AI-Educational-Suite

# Install dependencies
pip install -r requirements.txt

# Run program
python main.py
```

## 📱 Usage

1. **Launch Application**: `python main.py`
2. **Upload Study Materials**: Click "Add Document" → Select textbooks/notes
3. **Start Learning**: Ask questions, generate quizzes, create concept maps
   

## 🎓 Educational Capabilities

**Learning Tools**:
- Concept explanation with real-world examples
- Interactive concept mapping
- Homework hints (guidance, not answers)
- Step-by-step problem solving

**Assessment Features**:
- Custom quiz generation (1-20 questions)
- Adaptive practice sessions
- Timed assessments
- Comprehensive skill evaluation

**Content Processing**:
- PDF page selection and text extraction
- Audio file import and live recording
- Screen capture integration
- Smart clipboard monitoring

## 🔒 Privacy & Offline Operation

- **100% Local Processing**: All AI inference happens on your device
- **No Data Transmission**: Zero external API calls after model download
- **Complete Privacy**: Student data never leaves your computer
- **Offline Ready**: Works without internet after initial setup

## 🏗️ Architecture

```
├── ModelManager: Handles Gemma 3n model downloads and management
├── TokenManager: 32K context window optimization
├── DocumentViewer: Multi-format document processing
├── Educational Tools: Subject-specific tutoring functions
├── Multimodal Support: Image, audio, and text processing
└── Offline Storage: Local model caching and configuration
```

## 📊 Technical Specifications

- **Context Window**: 32,000 tokens with real-time usage tracking
- **Streaming**: Real-time response generation with TextStreamer
- **Memory Optimization**: 4-bit quantization for consumer hardware
- **Thread Safety**: Dedicated generation threads with stop controls
- **Format Support**: 15+ document formats, 7+ audio formats

## 🎯 Hackathon Inspiration

This project was made for The Gemma 3n Impact Challenge

## 📁 Project Structure

```
├── main.py                 # Core application with Gemma 3n integration
├── requirements.txt        # Dependencies
├── README.md              # This file
└── unsloth_compiled_cache/                # Local model storage (auto-created)
```

## License

CC BY 4.0
