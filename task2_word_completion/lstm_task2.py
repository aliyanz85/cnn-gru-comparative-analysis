

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
import pickle
import streamlit as st

class UltraOptimizedGRU(nn.Module):
    """
    Ultra-optimized GRU model for fast training on M1 MacBook Air
    """
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_layers=1):
        super(UltraOptimizedGRU, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded, hidden)
        gru_out = self.dropout(gru_out)
        output = self.fc(gru_out)
        return output, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, batch_size, self.hidden_dim).zero_()
        return hidden

class FastTextProcessor:
    """
    Optimized text preprocessing for speed
    """
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
    
    def preprocess_text(self, text):
        """Fast text cleaning and normalization"""
        text = text.lower()
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def build_vocab(self, text):
        """Build vocabulary from limited text sample"""
        words = text.split()
        word_counts = Counter(words)
        filtered_words = {word: count for word, count in word_counts.items() 
                         if count >= self.min_freq}
        
        self.word_to_idx = {word: idx + 1 for idx, word in enumerate(filtered_words.keys())}
        self.word_to_idx['<UNK>'] = 0
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        
        return self.vocab_size
    
    def text_to_sequences(self, text, seq_length=15):
        """Convert text to training sequences with limited length"""
        words = text.split()
        sequences = []
        targets = []
        
        for i in range(len(words) - seq_length):
            seq = [self.word_to_idx.get(word, 0) for word in words[i:i+seq_length]]
            target = self.word_to_idx.get(words[i+seq_length], 0)
            sequences.append(seq)
            targets.append(target)
        
        return sequences, targets

def load_limited_shakespeare_data(max_words=50000):
    """Load limited Shakespeare dataset for fast training"""
    try:
        with open('Shakespeare plays/alllines.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        words = text.split()
        limited_text = ' '.join(words[:max_words])
        
        print(f"Loaded {len(limited_text):,} characters from Shakespeare dataset")
        print(f"Limited to first {len(words[:max_words]):,} words")
        return limited_text
    except FileNotFoundError:
        print("Shakespeare dataset not found, using sample text")
        return """
        To be or not to be that is the question
        Whether tis nobler in the mind to suffer
        The slings and arrows of outrageous fortune
        Or to take arms against a sea of troubles
        And by opposing end them
        All the world's a stage and all the men and women merely players
        They have their exits and their entrances
        And one man in his time plays many parts
        Romeo Romeo wherefore art thou Romeo
        Deny thy father and refuse thy name
        Or if thou wilt not be but sworn my love
        And I'll no longer be a Capulet
        What light through yonder window breaks
        It is the east and Juliet is the sun
        Friends Romans countrymen lend me your ears
        I come to bury Caesar not to praise him
        The evil that men do lives after them
        The good is oft interred with their bones
        """

def train_model_fast(model, train_loader, val_loader, epochs=2, lr=0.001):
    """Ultra-fast training loop optimized for M1 MacBook Air"""
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(sequences)
            outputs = outputs[:, -1, :]  # Last timestep only
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs, _ = model(sequences)
                outputs = outputs[:, -1, :]
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return train_losses, val_losses

def plot_training_curves(train_losses, val_losses):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
    
    plt.title('Ultra-Optimized GRU Training Progress', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/ultra_optimized_gru_training.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_text(model, processor, seed_text, max_length=15, temperature=1.0):
    """Generate text continuation from seed"""
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.eval()
    
    words = seed_text.split()
    generated = seed_text
    
    with torch.no_grad():
        for _ in range(max_length):
            if len(words) >= 15:
                input_words = words[-15:]
            else:
                input_words = words
            
            input_indices = [processor.word_to_idx.get(word, 0) for word in input_words]
            
            if len(input_indices) < 15:
                input_indices = [0] * (15 - len(input_indices)) + input_indices
            else:
                input_indices = input_indices[-15:]
            
            input_tensor = torch.LongTensor([input_indices]).to(device)
            
            outputs, _ = model(input_tensor)
            last_output = outputs[0, -1, :]
            
            if temperature != 1.0:
                last_output = last_output / temperature
            
            probs = torch.softmax(last_output, dim=0)
            next_word_idx = torch.multinomial(probs, 1).item()
            next_word = processor.idx_to_word[next_word_idx]
            
            generated += " " + next_word
            words.append(next_word)
            
            if next_word in ['.', '!', '?']:
                break
    
    return generated

def predict_next_word(model, processor, input_text, top_k=5):
    """Predict next word with probabilities"""
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.eval()
    
    words = input_text.split()
    
    with torch.no_grad():
        if len(words) >= 15:
            input_words = words[-15:]
        else:
            input_words = words
        
        input_indices = [processor.word_to_idx.get(word, 0) for word in input_words]
        
        if len(input_indices) < 15:
            input_indices = [0] * (15 - len(input_indices)) + input_indices
        else:
            input_indices = input_indices[-15:]
        
        input_tensor = torch.LongTensor([input_indices]).to(device)
        
        outputs, _ = model(input_tensor)
        last_output = outputs[0, -1, :]
        
        top_probs, top_indices = torch.topk(last_output, top_k)
        top_probs = torch.softmax(top_probs, dim=0)
        
        predictions = []
        for i in range(top_k):
            word = processor.idx_to_word[top_indices[i].item()]
            prob = top_probs[i].item()
            predictions.append((word, prob))
        
        return predictions

def cli_interface(model, processor):
    """Simple command-line interface for word prediction"""
    print("\n" + "="*60)
    print("ULTRA-OPTIMIZED GRU WORD COMPLETION - CLI INTERFACE")
    print("="*60)
    print("Type a partial sentence and press Enter to get word suggestions")
    print("Type 'quit' to exit, 'generate' for text generation")
    print("-"*60)
    
    while True:
        user_input = input("\nEnter text: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'generate':
            seed = input("Enter seed text for generation: ").strip()
            if seed:
                generated = generate_text(model, processor, seed, max_length=15)
                print(f"Generated: {generated}")
        elif user_input:
            try:
                predictions = predict_next_word(model, processor, user_input, top_k=5)
                print("\nTop 5 word suggestions:")
                for i, (word, prob) in enumerate(predictions, 1):
                    print(f"{i}. {word} ({prob:.2%})")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Please enter some text")

def streamlit_interface(model, processor):
    """Streamlit web interface"""
    st.set_page_config(
        page_title="Ultra-Optimized GRU Word Completion",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("⚡ Ultra-Optimized GRU Word Completion")
    st.markdown("**Optimized for M1 MacBook Air - Under 20 minutes training**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("✍️ Word Prediction")
        
        input_text = st.text_area(
            "Enter your text:",
            value="To be or not to be",
            height=100
        )
        
        if st.button("🔮 Predict Next Word", type="primary"):
            if input_text.strip():
                try:
                    predictions = predict_next_word(model, processor, input_text, top_k=5)
                    
                    st.markdown("### 💡 Word Suggestions:")
                    cols = st.columns(5)
                    for i, (word, prob) in enumerate(predictions):
                        with cols[i]:
                            st.metric(word, f"{prob:.1%}")
                
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with col2:
        st.header("🎭 Text Generation")
        
        seed_text = st.text_input("Seed text:", value="To be or not to be")
        max_length = st.slider("Max length:", 5, 20, 12)
        temperature = st.slider("Temperature:", 0.1, 2.0, 1.0)
        
        if st.button("✨ Generate Text", type="secondary"):
            if seed_text.strip():
                try:
                    with st.spinner("Generating..."):
                        generated = generate_text(model, processor, seed_text, max_length, temperature)
                    st.text_area("Generated:", value=generated, height=100)
                except Exception as e:
                    st.error(f"Error: {e}")

def main():
    """Main training and inference pipeline - Ultra-optimized for speed"""
    print("ULTRA-OPTIMIZED GRU FOR SHAKESPEARE WORD COMPLETION")
    print("="*70)
    print("Optimized for M1 MacBook Air - Target: Under 20 minutes")
    print("Architecture: GRU(128) + Embedding(100) + FC")
    print("Dataset: First 50,000 words, Sequence: 15, Batch: 32, Epochs: 2")
    print("="*70)

    print("\n1. Loading limited Shakespeare dataset (50,000 words)...")
    text = load_limited_shakespeare_data(max_words=50000)
    
    processor = FastTextProcessor(min_freq=2)
    processed_text = processor.preprocess_text(text)
    vocab_size = processor.build_vocab(processed_text)
    
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Processed text: {len(processed_text.split()):,} words")
    
    print("\n2. Creating training sequences (length=15)...")
    sequences, targets = processor.text_to_sequences(processed_text, seq_length=15)
    print(f"Created {len(sequences):,} training sequences")
    
    train_size = int(0.8 * len(sequences))
    val_size = int(0.1 * len(sequences))
    
    train_sequences = sequences[:train_size]
    train_targets = targets[:train_size]
    val_sequences = sequences[train_size:train_size + val_size]
    val_targets = targets[train_size:train_size + val_size]
    
    print(f"Training samples: {len(train_sequences):,}")
    print(f"Validation samples: {len(val_sequences):,}")
    
    train_dataset = TensorDataset(torch.LongTensor(train_sequences), torch.LongTensor(train_targets))
    val_dataset = TensorDataset(torch.LongTensor(val_sequences), torch.LongTensor(val_targets))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print("\n3. Creating ultra-optimized GRU model...")
    model = UltraOptimizedGRU(vocab_size, embedding_dim=100, hidden_dim=128, num_layers=1)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n4. Training model (2 epochs)...")
    import time
    start_time = time.time()
    
    train_losses, val_losses = train_model_fast(model, train_loader, val_loader, epochs=2, lr=0.001)
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    print("\n5. Plotting training curves...")
    plot_training_curves(train_losses, val_losses)
    
    print("\n6. Generating sample completions...")
    sample_seeds = [
        "To be or not to be",
        "All the world's a stage",
        "Romeo Romeo wherefore art thou",
        "What light through yonder window",
        "Friends Romans countrymen"
    ]
    
    print("\nSample Text Completions:")
    print("-" * 50)
    
    for seed in sample_seeds:
        generated = generate_text(model, processor, seed, max_length=12)
        print(f"Seed: '{seed}'")
        print(f"Generated: {generated}")
        print()
    
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/ultra_optimized_gru.pth')
    with open('models/processor.pkl', 'wb') as f:
        pickle.dump(processor, f)
    
    print("Model saved to models/ultra_optimized_gru.pth")
    print("Processor saved to models/processor.pkl")
    
    os.makedirs('results', exist_ok=True)
    with open('results/ultra_optimized_results.txt', 'w') as f:
        f.write("Ultra-Optimized GRU Word Completion Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)\n")
        f.write(f"Vocabulary Size: {vocab_size:,}\n")
        f.write(f"Training Sequences: {len(train_sequences):,}\n")
        f.write(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"Architecture: GRU(128) + Embedding(100) + FC\n")
        f.write(f"Sequence Length: 15, Batch Size: 32, Epochs: 2\n\n")
        
        f.write("Training Progress:\n")
        for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
            f.write(f"Epoch {i}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}\n")
        
        f.write("\nSample Completions:\n")
        for seed in sample_seeds:
            generated = generate_text(model, processor, seed, max_length=12)
            f.write(f"'{seed}' -> '{generated}'\n")
    
    print("\n" + "="*70)
    print("ULTRA-OPTIMIZED TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Total time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print("="*70)
    
    print("\nChoose interface:")
    print("1. Command Line Interface")
    print("2. Streamlit Web Interface")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        cli_interface(model, processor)
    elif choice == "2":
        print("\nTo run Streamlit interface:")
        print("streamlit run ultra_optimized_lstm_task2.py")
    else:
        print("Exiting...")

if __name__ == "__main__":
    main()
