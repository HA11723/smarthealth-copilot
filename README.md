# 🧠 SmartHealth Copilot: AI-Powered Medical Assistant

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-red.svg)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5--turbo-green.svg)](https://openai.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent medical assistant that combines Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) to provide preliminary medical insights and symptom analysis. Built with modern AI technologies and designed for educational purposes.

## 🌟 Features

- **🤖 AI-Powered Analysis**: Advanced symptom analysis using OpenAI's GPT-3.5-turbo
- **🔍 Similar Case Retrieval**: Find similar medical cases using vector database technology
- **⚡ Urgency Assessment**: Real-time evaluation of symptom urgency levels
- **🌍 Multilingual Support**: Input and output in multiple languages (English, Arabic, Chinese, Japanese, Korean, Hindi)
- **📊 Medical History Processing**: Comprehensive analysis of medical histories
- **🎯 Diagnostic Suggestions**: AI-generated diagnostic possibilities with confidence levels
- **💻 User-Friendly Interface**: Clean, intuitive Streamlit web interface
- **🔒 Medical Disclaimers**: Built-in safety features and appropriate medical disclaimers

## 🛠️ Tech Stack

### Core Technologies

- **Python 3.8+** - Primary programming language
- **Streamlit** - Web application framework
- **OpenAI API** - Large Language Model integration
- **LangChain** - LLM orchestration and RAG implementation

### Database & Storage

- **Pinecone** - Vector database for similarity search
- **ChromaDB** - Alternative vector database option
- **JSON** - Local data storage for medical cases

### AI/ML Libraries

- **Sentence Transformers** - Text embedding generation
- **Transformers** - Hugging Face transformers library
- **Pandas & NumPy** - Data processing and manipulation

### Development Tools

- **FastAPI** - Backend API framework (optional)
- **Pytest** - Testing framework
- **Black & Flake8** - Code formatting and linting

## 📁 Project Structure

```
smarthealth-copilot/
├── app/
│   ├── main.py              # Streamlit main application
│   ├── core/                # Core application logic
│   │   ├── llm.py          # LLM integration
│   │   ├── vector_db.py    # Vector database operations
│   │   ├── retrieval.py    # RAG retrieval logic
│   │   └── medical_data.py # Medical data processing
│   └── data/               # Data files
│       ├── medical_cases.json
│       └── symptoms_db.json
├── tests/                  # Test files
├── requirements.txt         # Python dependencies
├── setup.py               # Package setup
├── .env.example           # Environment variables template
├── .gitignore             # Git ignore file
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- (Optional) Pinecone API key for vector database

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/smarthealth-copilot.git
   cd smarthealth-copilot
   ```

2. **Create and activate virtual environment**

   ```bash
   python -m venv venv

   # On macOS/Linux:
   source venv/bin/activate

   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` file with your API keys:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_ENVIRONMENT=your_pinecone_environment
   ```

5. **Run the application**

   ```bash
   streamlit run app/main.py
   ```

6. **Access the application**
   - Open your browser and go to `http://localhost:8501`
   - The application will be available in your default browser

## 📖 Usage Guide

### Basic Usage

1. **Enter Patient Information**: Provide age, gender, and primary symptoms
2. **Add Additional Symptoms**: Include any additional symptoms or medical history
3. **Get AI Analysis**: Receive comprehensive medical analysis including:
   - Possible diagnoses with confidence levels
   - Urgency assessment
   - Similar medical cases
   - Medical recommendations

### Multilingual Support

The application supports multiple languages:

- **English**: Default language
- **Arabic**: العربية
- **Chinese**: 中文
- **Japanese**: 日本語
- **Korean**: 한국어
- **Hindi**: हिन्दी

Simply input your symptoms in any supported language, and the AI will respond in the same language.

### Example Inputs

**English:**

```
Age: 25
Gender: Female
Symptoms: Headache, fever, fatigue
```

**Arabic:**

```
العمر: 25
الجنس: أنثى
الأعراض: صداع، حمى، إرهاق
```

## 🔧 Configuration

### Environment Variables

| Variable               | Description                                  | Required |
| ---------------------- | -------------------------------------------- | -------- |
| `OPENAI_API_KEY`       | Your OpenAI API key                          | Yes      |
| `PINECONE_API_KEY`     | Pinecone API key for vector DB               | No       |
| `PINECONE_ENVIRONMENT` | Pinecone environment                         | No       |
| `OPENAI_MODEL`         | OpenAI model to use (default: gpt-3.5-turbo) | No       |

### Customization

You can customize the application by:

- Modifying medical data in `app/data/`
- Adjusting AI prompts in `app/main.py`
- Adding new languages in the language detection function
- Customizing the UI in the Streamlit interface

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Add tests** for new functionality
5. **Run tests** to ensure everything works
   ```bash
   pytest tests/
   ```
6. **Commit your changes**
   ```bash
   git commit -m "Add: your feature description"
   ```
7. **Push to your branch**
   ```bash
   git push origin feature/your-feature-name
   ```
8. **Create a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include type hints for function parameters
- Write tests for new features
- Update documentation as needed


## ⚠️ Important Disclaimer

**This application is for educational and research purposes only. It is not intended to provide medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.**

- The AI-generated analysis should not replace professional medical consultation
- Use this tool as a supplementary resource only
- For medical emergencies, contact emergency services immediately
- The developers are not responsible for any medical decisions made based on this application

## 🙏 Acknowledgments

- OpenAI for providing the GPT API
- Streamlit for the web framework
- The open-source community for various libraries and tools
- Medical professionals who provided guidance on appropriate disclaimers

## 📞 Support

If you encounter any issues or have questions:

1. **Check the Issues** page for existing solutions
2. **Create a new Issue** with detailed information
3. **Include error messages** and steps to reproduce
4. **Provide your environment details** (OS, Python version, etc.)

## 🔗 Links

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)

---

**Made with ❤️ for the AI and healthcare communities**
