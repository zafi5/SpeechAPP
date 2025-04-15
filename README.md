# Speech App

A web application that provides both **Speech-to-Text (STT)** and **Text-to-Speech (TTS)** capabilities.

## Requirements

To run the Speech App locally, please make sure your environment meets the following prerequisites:

- **Python 3.9 or lower** is required to work with the **DeepSpeech** model.
- **DeepSpeech model file** (`deepspeech-0.9.3-models.pbmm`) must be downloaded manually.
- A **virtual environment** is recommended for isolating dependencies.

## Installation Guide

### 1. Install Python 3.9 or Lower

You need to use Python version 3.9 or below to work with DeepSpeech (`deepspeech-0.9.3` version). You can use `pyenv` or another version manager to install Python 3.9.

To install Python 3.9, follow the instructions based on your operating system:

- **macOS**: [Install Python using Homebrew](https://brew.sh/)
- **Windows**: [Download Python 3.9 from the official website](https://www.python.org/downloads/release/python-3910/)
- **Linux**: Use `apt` or `yum` to install Python 3.9 based on your distribution.

### 2. Set Up a Virtual Environment

```bash
# Create a virtual environment
python3.9 -m venv venv

# Activate the virtual environment
# For macOS/Linux:
source venv/bin/activate

# For Windows:
venv\Scripts\activate
```

3. Install Dependencies
   Once your virtual environment is activated, install the necessary dependencies.

# Install the required Python packages

```bash
pip install -r requirements.txt
```

4. Download DeepSpeech Model

   - The DeepSpeech 0.9.3 model (deepspeech-0.9.3-models.pbmm) is required for the speech-to-text functionality. You need to download this model manually.

   - Download the Deepspeech 0.9.3 model from the official DeepSpeech GitHub repository:

   - [Deepspeech 0.9.3 Model](https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3)

   - After downloading, place the model file (deepspeech-0.9.3-models.pbmm) in the root directory of your project or specify the path to the model in your code.

5. Running the Application
   Once everything is installed, and the model is downloaded, run the Django development server.

# Run the server

```bash
python manage.py runserver
```

The application will be available at http://127.0.0.1:8000/.

6. Additional Setup for Text-to-Speech (TTS)
   To use the Text-to-Speech functionality, you can use Hugging Face’s microsoft/speecht5_tts model. Make sure to install the transformers package if you haven't already:

```bash
pip install transformers
```

You may also need additional dependencies, such as sentencepiece, which is required by the SpeechT5 tokenizer

```bash
pip install sentencepiece
```
