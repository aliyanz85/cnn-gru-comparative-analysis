# CNN-GRU Comparative Analysis

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-green.svg)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.12-blue.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comparative study of CNN-based deep learning and traditional feature extraction pipelines for signature recognition, paired with an LSTM-based text generation model trained on Shakespeare's works.

---

## Highlights

- **Deep CNN** for signature classification achieving **94.2% accuracy**
- **HOG + SVM** and **SIFT + SVM** baselines for comparative benchmarking
- **LSTM text generation** with perplexity of **12.3** on Shakespeare corpus
- **Interactive Streamlit UI** for real-time word completion and text generation

---

## Project Structure

```
├── task1_signature_recognition/     # CNN and feature extraction pipelines
├── task2_word_completion/           # LSTM model and Streamlit interface
├── report/                          # Technical report (IEEE format)
├── data/                            # Dataset storage
├── models/                          # Trained model weights
├── results/                         # Evaluation outputs and visualizations
├── requirements.txt                 # Python dependencies
├── run_assignment.py                # Main execution script
└── create_zip.py                    # ZIP bundle creation
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scriptsctivate
pip install -r requirements.txt
```

### Usage

**Run full pipeline:**
```bash
python run_assignment.py
```

**Run individual tasks:**
```bash
# Signature Recognition (CNN vs. HOG vs. SIFT)
python run_assignment.py --task 1

# Text Generation (LSTM)
python run_assignment.py --task 2
```

**Launch the interactive text generation UI:**
```bash
cd task2_word_completion
streamlit run streamlit_app.py
```

---

## Task 1: Signature Recognition

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| CNN | **94.2%** | 0.94 | 0.94 | 0.94 |
| HOG + SVM | 87.1% | 0.87 | 0.86 | 0.86 |
| SIFT + SVM | 82.5% | 0.83 | 0.82 | 0.82 |

The deep CNN significantly outperforms traditional handcrafted feature pipelines, validating end-to-end learned representations for biometric signature verification.

---

## Task 2: Word Completion (LSTM)

- Trained on Shakespeare's complete works
- Temperature-controlled text generation
- Real-time suggestions via Streamlit interface
- Perplexity: **12.3**

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Deep Learning | TensorFlow 2.20, Keras 3.11 |
| Classical ML | scikit-learn 1.7 |
| Computer Vision | OpenCV 4.12 |
| UI | Streamlit 1.50 |
| Report | LaTeX (IEEE format) |

---

## Results

Detailed evaluation metrics, confusion matrices, and visualizations are available in the `results/` directory.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
