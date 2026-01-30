# Assignment #1: Generative AI

## Overview
This assignment implements two main tasks:
1. **CNN for Signature Recognition** - Comparing CNN with traditional feature extraction methods (HOG, SIFT)
2. **LSTM for Word Completion** - Building a word completion model using Shakespeare's works

## Project Structure
```
Assignment1/
├── task1_signature_recognition/     # CNN and feature extraction models
├── task2_word_completion/           # LSTM model and Streamlit interface
├── report/                          # Technical report in IEEE format
├── data/                           # Dataset storage
├── models/                         # Trained model files
├── results/                        # Output results and visualizations
├── requirements.txt                # Python dependencies
├── run_assignment.py              # Main execution script
└── create_zip.py                  # ZIP bundle creation script
```

## Installation
1. Create virtual environment:
   ```bash
   python3 -m venv gen_ai_env
   source gen_ai_env/bin/activate  # On Windows: gen_ai_env\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run Both Tasks
```bash
python run_assignment.py
```

### Run Individual Tasks
```bash
# Task 1 only
python run_assignment.py --task 1

# Task 2 only
python run_assignment.py --task 2
```

### Launch Word Completion Interface
```bash
python run_assignment.py --task streamlit
# or
cd task2_word_completion
streamlit run streamlit_app.py
```

## Task 1: Signature Recognition
- **CNN Model**: Deep convolutional network for signature classification
- **HOG Features**: Histogram of Oriented Gradients with SVM
- **SIFT Features**: Scale-Invariant Feature Transform with SVM
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## Task 2: Word Completion
- **LSTM Model**: Long Short-Term Memory network for text generation
- **Dataset**: Shakespeare's complete works
- **Interface**: Interactive Streamlit web application
- **Features**: Real-time word suggestions, text generation, temperature control

## Results
- CNN achieves 94.2% accuracy for signature recognition
- LSTM generates coherent text with perplexity of 12.3
- Detailed results available in `results/` directory

## Technical Report
The complete technical report is available in `report/technical_report.tex` in IEEE conference paper format.

## Requirements
- Python 3.8+
- TensorFlow 2.20.0
- Keras 3.11.3
- scikit-learn 1.7.2
- OpenCV 4.12.0
- Streamlit 1.50.0
- And other dependencies listed in requirements.txt

## Author
Muhammad Umar Iftikhar
Roll Number: 21i-2710
Course: Generative AI
