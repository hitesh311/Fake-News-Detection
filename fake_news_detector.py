import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm.auto import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)  # Changed 'label' to 'labels' for compatibility
        }

def clean_text(text):
    """Basic text cleaning"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def preprocess_data(df):
    """Preprocess the dataframe"""
    # Combine title and text
    df['content'] = df['title'] + " " + df['text']
    
    # Clean text
    df['clean_content'] = df['content'].apply(clean_text)
    
    # Remove empty texts
    df = df[df['clean_content'].str.len() > 10]
    
    return df

def load_data(use_augmented=True):
    """Load and prepare the dataset"""
    if use_augmented and os.path.exists('augmented_fake_news_data.csv'):
        print("Loading augmented dataset...")
        df = pd.read_csv('augmented_fake_news_data.csv')
        # Ensure we have the expected columns
        if 'clean_content' not in df.columns:
            df = preprocess_data(df)
    else:
        print("Loading and preprocessing original dataset...")
        # Load original datasets
        fake_df = pd.read_csv('Fake.csv')
        fake_df['label'] = 0  # 0 for fake news
        
        true_df = pd.read_csv('True.csv')
        true_df['label'] = 1  # 1 for true news
        
        # Combine and preprocess
        df = pd.concat([fake_df, true_df], ignore_index=True)
        df = preprocess_data(df)
    
    return df

def train_model(train_dataset, val_dataset, epochs=3):
    """Train the model using a custom training loop"""
    print("\nInitializing model...")
    
    # Load pre-trained model and tokenizer
    model_name = 'distilbert-base-uncased'  # Using DistilBERT for faster training
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Fake", 1: "True"},
        label2id={"Fake": 0, "True": 1}
    ).to(device)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        # Training phase
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_train_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Evaluation phase
        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        
        for batch in tqdm(val_loader, desc="Evaluating"):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                total_eval_loss += loss.item()
                
                # Calculate accuracy
                preds = torch.argmax(logits, dim=1)
                total_eval_accuracy += (preds == labels).sum().item()
        
        # Calculate average validation loss and accuracy
        avg_val_loss = total_eval_loss / len(val_loader)
        avg_val_accuracy = total_eval_accuracy / len(val_loader.dataset)
        
        print(f"\nEpoch {epoch + 1}:")
        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        print(f"  Validation Accuracy: {avg_val_accuracy:.4f}")
    
    return model, tokenizer

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data(use_augmented=True)
    
    # Split data into train and validation sets
    train_df, val_df = train_test_split(
        df[['clean_content', 'label']],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    
    print("\nDataset sizes:")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = NewsDataset(
        train_df['clean_content'].values,
        train_df['label'].values,
        tokenizer
    )
    
    val_dataset = NewsDataset(
        val_df['clean_content'].values,
        val_df['label'].values,
        tokenizer
    )
    
    # Train the model
    model, tokenizer = train_model(train_dataset, val_dataset, epochs=3)
    
    # Create output directory
    output_dir = './model_save'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model and tokenizer
    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save the complete model with additional metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.config.to_dict(),
        'tokenizer': tokenizer,
        'label2id': {"Fake": 0, "True": 1},
        'id2label': {0: "Fake", 1: "True"}
    }, os.path.join(output_dir, 'fake_news_model.pt'))
    
    print(f"\nModel and tokenizer saved to {output_dir}")
    print("\nTraining complete!")

if __name__ == "__main__":
    main()
