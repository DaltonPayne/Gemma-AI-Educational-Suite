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
git clone https://github.com/yourusername/Gemma-AI-Educational-Suite.git
cd Gemma-AI-Educational-Suite

# Install dependencies
pip install -r requirements.txt

# Run program
python main.py


## 📱 Usage

1. **Launch Application**: `python main.py`
2. **Download AI Model**: Click "🤖 Model Manager" → "📥 Download Model" → Select Gemma 3n variant
3. **Enable Offline Mode**: Check "🌐 Offline Mode" for complete privacy
4. **Upload Study Materials**: Click "Add Document" → Select textbooks/notes
5. **Start Learning**: Ask questions, generate quizzes, create concept maps

## 🧠 Gemma 3n Model Variants

| Model | Size | RAM Required | Use Case |
|-------|------|-------------|----------|
| `gemma-3n-2B-it` | ~1.5GB | 4GB+ | Fast responses, basic tutoring |
| `gemma-3n-E4B-it` | ~2.5GB | 6GB+ | **Recommended** - balanced performance |
| `gemma-3n-5B-it` | ~3.5GB | 8GB+ | Higher quality responses |
| `gemma-3n-8B-it` | ~5GB | 12GB+ | Best quality, detailed explanations |

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

## 🎯 Hackathon Innovation

This project demonstrates:
1. **Practical Gemma 3n Integration**: Real-world educational application
2. **Complete Offline Capability**: Addresses connectivity and privacy concerns
3. **Multimodal Learning**: Beyond text-only AI interactions
4. **Educational Focus**: Purpose-built for learning enhancement
5. **User Experience**: Intuitive GUI with comprehensive features

## 📁 Project Structure

```
├── main.py                 # Core application with Gemma 3n integration
├── requirements.txt        # Dependencies
├── README.md              # This file
└── models/                # Local model storage (auto-created)
```

## License

CC BY 4.0

---

**🤖 Powered by Gemma 3n** | **🌐 Works Completely Offline** | **🎓 Educational AI Revolution**
