#!/usr/bin/env python3


import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
from skimage.feature import hog
import cv2

class SimpleCNN(nn.Module):
    """
    Simple CNN for signature recognition
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Calculate the size after convolutions and pooling
        # Assuming input size 128x128
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Convolutional layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)
        
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        
        x = self.pool(self.relu(self.conv3(x)))
        x = self.dropout(x)
        
        x = self.pool(self.relu(self.conv4(x)))
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def create_signature_images(data, labels, image_size=(128, 128)):
    """
    Convert 1D feature data to 2D images for CNN
    """
    images = []
    for i in range(len(data)):
        # Reshape 1D features to 2D image
        feature_vector = data[i]
        # Pad or truncate to image_size
        if len(feature_vector) < image_size[0] * image_size[1]:
            # Pad with zeros
            padded = np.pad(feature_vector, (0, image_size[0] * image_size[1] - len(feature_vector)))
        else:
            # Truncate
            padded = feature_vector[:image_size[0] * image_size[1]]
        
        # Reshape to 2D
        image = padded.reshape(image_size[0], image_size[1])
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        images.append(image)
    
    return np.array(images)

def extract_hog_features(images):
    """
    Extract HOG features from images
    """
    hog_features = []
    for image in images:
        # Convert to uint8
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Extract HOG features
        features = hog(image_uint8, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys')
        hog_features.append(features)
    
    return np.array(hog_features)

def extract_sift_features(images):
    """
    Extract SIFT features from images
    """
    sift = cv2.SIFT_create()
    sift_features = []
    
    for image in images:
        image_uint8 = (image * 255).astype(np.uint8)
        keypoints, descriptors = sift.detectAndCompute(image_uint8, None)
        
        if descriptors is not None:
            mean_desc = np.mean(descriptors, axis=0)
            std_desc = np.std(descriptors, axis=0)
            features = np.concatenate([mean_desc, std_desc])
        else:
            features = np.zeros(256)  # 128 * 2 for mean and std
        sift_features.append(features)
    
    return np.array(sift_features)

def train_cnn(model, train_loader, val_loader, epochs=20, learning_rate=0.001):
    """
    Train the CNN model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }

def evaluate_cnn(model, test_loader):
    """
    Evaluate the CNN model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels)

def plot_training_history(history, save_path='results/cnn_training_history.png'):
    """
    Plot training and validation curves
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history['train_accuracies'], label='Training Accuracy')
    ax1.plot(history['val_accuracies'], label='Validation Accuracy')
    ax1.set_title('CNN Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_losses'], label='Training Loss')
    ax2.plot(history['val_losses'], label='Validation Loss')
    ax2.set_title('CNN Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function for proper Task 1 implementation
    """
    print("PROPER TASK 1: CNN FOR SIGNATURE RECOGNITION")
    print("="*60)
    print("Using actual CNN implementation with PyTorch")
    
    print("\nCreating synthetic signature dataset...")
    X, y = make_classification(n_samples=2000, n_features=100, n_classes=10, 
                              n_informative=50, random_state=42)
    
    class_names = [f'Person_{i+1}' for i in range(10)]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print("\nConverting features to images...")
    train_images = create_signature_images(X_train, y_train)
    val_images = create_signature_images(X_val, y_val)
    test_images = create_signature_images(X_test, y_test)
    train_images = train_images.reshape(-1, 1, 128, 128)
    val_images = val_images.reshape(-1, 1, 128, 128)
    test_images = test_images.reshape(-1, 1, 128, 128)
    train_dataset = TensorDataset(torch.FloatTensor(train_images), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(val_images), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(test_images), torch.LongTensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print("\nTraining CNN model...")
    cnn_model = SimpleCNN(num_classes=10)
    history = train_cnn(cnn_model, train_loader, val_loader, epochs=20)
    plot_training_history(history)
    print("\nEvaluating CNN model...")
    cnn_pred, cnn_true = evaluate_cnn(cnn_model, test_loader)
    
    cnn_accuracy = accuracy_score(cnn_true, cnn_pred)
    cnn_precision = precision_score(cnn_true, cnn_pred, average='weighted')
    cnn_recall = recall_score(cnn_true, cnn_pred, average='weighted')
    cnn_f1 = f1_score(cnn_true, cnn_pred, average='weighted')
    
    print(f"CNN Accuracy: {cnn_accuracy:.4f}")
    print(f"CNN Precision: {cnn_precision:.4f}")
    print(f"CNN Recall: {cnn_recall:.4f}")
    print(f"CNN F1-Score: {cnn_f1:.4f}")
    
    print("\nExtracting HOG features...")
    train_hog = extract_hog_features(train_images.reshape(-1, 128, 128))
    test_hog = extract_hog_features(test_images.reshape(-1, 128, 128))
    
    print("\nExtracting SIFT features...")
    train_sift = extract_sift_features(train_images.reshape(-1, 128, 128))
    test_sift = extract_sift_features(test_images.reshape(-1, 128, 128))
    
    print("\nTraining HOG + SVM...")
    hog_svm = SVC(kernel='rbf', random_state=42)
    hog_svm.fit(train_hog, y_train)
    hog_pred = hog_svm.predict(test_hog)
    hog_accuracy = accuracy_score(y_test, hog_pred)
    
    print("\nTraining SIFT + SVM...")
    sift_svm = SVC(kernel='rbf', random_state=42)
    sift_svm.fit(train_sift, y_train)
    sift_pred = sift_svm.predict(test_sift)
    sift_accuracy = accuracy_score(y_test, sift_pred)
    
    print("\nTraining HOG + Logistic Regression...")
    hog_lr = LogisticRegression(random_state=42, max_iter=1000)
    hog_lr.fit(train_hog, y_train)
    hog_lr_pred = hog_lr.predict(test_hog)
    hog_lr_accuracy = accuracy_score(y_test, hog_lr_pred)
    
    results = {
        'CNN': {
            'accuracy': cnn_accuracy,
            'precision': cnn_precision,
            'recall': cnn_recall,
            'f1_score': cnn_f1,
            'confusion_matrix': confusion_matrix(y_test, cnn_pred)
        },
        'HOG + SVM': {
            'accuracy': hog_accuracy,
            'confusion_matrix': confusion_matrix(y_test, hog_pred)
        },
        'SIFT + SVM': {
            'accuracy': sift_accuracy,
            'confusion_matrix': confusion_matrix(y_test, sift_pred)
        },
        'HOG + LR': {
            'accuracy': hog_lr_accuracy,
            'confusion_matrix': confusion_matrix(y_test, hog_lr_pred)
        }
    }
    
    print("\n" + "="*60)
    print("FINAL RESULTS COMPARISON")
    print("="*60)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        if 'precision' in metrics:
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
    
    print("\nCreating visualizations...")
    plt.figure(figsize=(12, 6))
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    
    bars = plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    plt.title('Model Performance Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/proper_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, (model_name, metrics) in enumerate(results.items()):
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Blues', ax=axes[i],
                   xticklabels=class_names,
                   yticklabels=class_names)
        axes[i].set_title(f'{model_name} Confusion Matrix')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('results/proper_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    os.makedirs('results', exist_ok=True)
    with open('results/proper_task1_results.txt', 'w') as f:
        f.write("Proper Task 1: CNN for Signature Recognition - Results\n")
        f.write("=" * 60 + "\n\n")
        
        for model_name, metrics in results.items():
            f.write(f"{model_name}:\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            if 'precision' in metrics:
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n")
            f.write("\n")
    
    print("\n" + "="*60)
    print("PROPER TASK 1 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("This implementation uses actual CNN with PyTorch")
    print("Results saved to 'results/' directory")

if __name__ == "__main__":
    main()
