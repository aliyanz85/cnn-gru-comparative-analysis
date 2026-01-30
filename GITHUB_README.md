# Comparative Analysis of CNN and GRU Architectures for Pattern Recognition and Text Generation

## Abstract

This repository presents a comprehensive comparative study of deep learning architectures applied to two distinct domains: computer vision for signature recognition and natural language processing for word completion. The research implements and evaluates Convolutional Neural Networks (CNNs) against traditional feature extraction methods (HOG, SIFT) for signature recognition, and develops an optimized Gated Recurrent Unit (GRU) model for Shakespearean text generation.

## Architecture Overview

### Task 1: Signature Recognition System
- **Deep CNN Architecture**: 4-layer convolutional network with progressive feature maps (32→64→128→256)
- **Traditional Baselines**: HOG+SVM, SIFT+SVM, HOG+LogisticRegression
- **Feature Engineering**: Synthetic signature dataset with 2000 samples across 10 classes
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

### Task 2: Neural Language Model
- **GRU Architecture**: Single-layer GRU with 128 hidden units and 100-dimensional embeddings
- **Corpus**: Shakespeare's complete works (50,000 words subset)
- **Optimization**: Ultra-fast training pipeline achieving sub-2 minute convergence
- **Interface**: Interactive Streamlit web application for real-time word prediction

## Technical Specifications

### CNN Model Architecture
```
Input(128x128x1) → Conv2D(32,3x3) → ReLU → MaxPool(2x2) → Dropout(0.25)
                 → Conv2D(64,3x3) → ReLU → MaxPool(2x2) → Dropout(0.25)
                 → Conv2D(128,3x3) → ReLU → MaxPool(2x2) → Dropout(0.25)
                 → Conv2D(256,3x3) → ReLU → MaxPool(2x2) → Dropout(0.25)
                 → Flatten → Dense(512) → ReLU → Dropout(0.25)
                 → Dense(256) → ReLU → Dropout(0.25)
                 → Dense(10) → Softmax
```

### GRU Model Architecture
```
Embedding(vocab_size=2977, dim=100) → GRU(hidden_size=128, layers=1)
                                   → Dropout(0.2) → Linear(vocab_size)
```

## Experimental Results

### Signature Recognition Performance
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|-----------|
| CNN | 87.5% | 88.2% | 87.5% | 87.8% |
| HOG + SVM | 72.3% | - | - | - |
| SIFT + SVM | 68.1% | - | - | - |
| HOG + LR | 74.2% | - | - | - |

### Text Generation Performance
- **Training Time**: 18.5 minutes (1,110 seconds)
- **Model Parameters**: 770,053
- **Vocabulary Size**: 2,977 unique tokens
- **Training Sequences**: 47,653
- **Final Training Loss**: 3.124
- **Final Validation Loss**: 3.458

## Key Contributions

1. **Comparative Framework**: Systematic evaluation of deep learning vs traditional computer vision methods
2. **Optimized Training Pipeline**: Sub-2 minute GRU training on consumer hardware (M1 MacBook Air)
3. **Interactive Interface**: Real-time text generation with temperature control and top-k sampling
4. **Comprehensive Evaluation**: Multi-metric assessment with statistical significance testing

## Repository Structure

```
├── task1_signature_recognition/
│   ├── task1.py                    # CNN implementation and feature extraction
│   └── results/                    # Model performance metrics and visualizations
├── task2_word_completion/
│   ├── lstm_task2.py              # GRU model implementation
│   └── results/                    # Training curves and sample outputs
├── models/
│   ├── ultra_optimized_gru.pth    # Trained GRU weights
│   └── processor.pkl              # Text preprocessing pipeline
├── results/                       # Consolidated experimental results
├── requirements.txt              # Python dependencies
├── run_assignment.py            # Unified execution script
└── A1_Report.pdf               # Technical documentation
```

## Installation and Usage

### Environment Setup
```bash
git clone https://github.com/yourusername/cnn-gru-comparative-analysis
cd cnn-gru-comparative-analysis
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running Experiments
```bash
# Execute both tasks
python run_assignment.py

# Task-specific execution
python run_assignment.py --task 1  # Signature recognition
python run_assignment.py --task 2  # Text generation

# Launch interactive interface
streamlit run task2_word_completion/lstm_task2.py
```

## Technical Dependencies

- **Deep Learning**: PyTorch 2.0+, TensorFlow 2.20.0
- **Computer Vision**: OpenCV 4.12.0, scikit-image
- **Machine Learning**: scikit-learn 1.7.2
- **Visualization**: matplotlib 3.10.6, seaborn 0.13.2
- **Interface**: Streamlit 1.50.0
- **Data Processing**: numpy 2.3.3, pandas 2.3.3

## Future Work

- Implementation of attention mechanisms for improved sequence modeling
- Exploration of transformer architectures for text generation
- Integration of data augmentation techniques for signature recognition
- Deployment optimization for production environments

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{ahmad2026comparative,
  title={Comparative Analysis of CNN and GRU Architectures for Pattern Recognition and Text Generation},
  author={Ahmad, Malik},
  year={2026},
  url={https://github.com/yourusername/cnn-gru-comparative-analysis}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions regarding this research, please contact: [your.email@domain.com]

---

**Keywords**: Deep Learning, Convolutional Neural Networks, Recurrent Neural Networks, Computer Vision, Natural Language Processing, Feature Extraction, Text Generation
