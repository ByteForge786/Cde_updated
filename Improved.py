import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
import os
import logging
from sklearn.metrics import accuracy_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SQLDataset(Dataset):
    def __init__(self, sql_names, sql_queries, attribute_lists, labels, tokenizer, max_length=512):
        self.sql_names = sql_names
        self.sql_queries = sql_queries
        self.attribute_lists = attribute_lists
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sql_names)

    def __getitem__(self, idx):
        name = self.sql_names[idx]
        query = self.sql_queries[idx]
        attributes = self.attribute_lists[idx]
        label = self.labels[idx]

        input_text = (
            "You are an SQL assistant tasked with analyzing SQL queries and identifying relevant attributes. "
            "Based on the SQL name and query provided, determine which attribute(s) from the given list are being "
            "analyzed, checked, or are most relevant to the query's purpose.\n\n"
            f"SQL Name: {name}\n"
            f"SQL Query: {query}\n"
            f"Pool of Attributes: {attributes}\n\n"
            "Identify the specific attribute(s) from the pool that are most relevant to this SQL query. "
            "The attributes you select should directly relate to the main purpose or focus of the query, "
            "as indicated by the SQL name and the operations performed in the query. "
            "Your answer should be one or more attributes from the provided pool, separated by commas if multiple."
        )
        
        target_text = label  # label is already a string (single label or comma-separated)

        input_encoding = self.tokenizer(input_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        target_encoding = self.tokenizer(target_text, max_length=100, padding='max_length', truncation=True, return_tensors="pt")

        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

def compute_accuracy(predictions, labels):
    # Flatten the predictions and labels
    pred_texts = [pred.strip() for pred in predictions]
    label_texts = [label.strip() for label in labels]
    
    # Calculate exact match accuracy
    return accuracy_score(label_texts, pred_texts)

def train_model(model, train_dataloader, val_dataloader, epochs=20, patience=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = 'best_model.pth'

    for epoch in range(epochs):
        logging.info(f'Epoch {epoch+1}/{epochs} - Training...')
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Store predictions and labels for accuracy calculation
            preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=100)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            train_preds.extend(decoded_preds)
            train_labels.extend(decoded_labels)

        # Validation
        logging.info(f'Epoch {epoch+1}/{epochs} - Validation...')
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

                # Store predictions and labels for accuracy calculation
                preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=100)
                decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                val_preds.extend(decoded_preds)
                val_labels.extend(decoded_labels)

        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        train_accuracy = compute_accuracy(train_preds, train_labels)
        val_accuracy = compute_accuracy(val_preds, val_labels)

        logging.info(f'Training Loss: {avg_train_loss}')
        logging.info(f'Validation Loss: {avg_val_loss}')
        logging.info(f'Training Accuracy: {train_accuracy}')
        logging.info(f'Validation Accuracy: {val_accuracy}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved with validation loss: {best_val_loss}")
        else:
            patience_counter += 1
            logging.info(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Load the best model
    model.load_state_dict(torch.load(best_model_path))
    return model

def predict(model, dataloader, tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=100)
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(decoded_preds)

    return predictions

def main():
    # Load and preprocess your data
    logging.info('Loading data...')
    data = pd.read_csv('your_data.csv')  # Adjust the filename as needed
    sql_names = data['sql_name'].tolist()
    sql_queries = data['SQL_query'].tolist()
    attribute_lists = data['attribute_list'].tolist()
    labels = data['label'].tolist()  # This column contains either single labels or comma-separated labels

    # Split the data
    logging.info('Splitting data...')
    train_names, val_names, train_queries, val_queries, train_attrs, val_attrs, train_labels, val_labels = train_test_split(
        sql_names, sql_queries, attribute_lists, labels, test_size=0.2, random_state=42
    )

    # Initialize tokenizer and model
    logging.info('Initializing tokenizer and model...')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')

    # Create datasets
    train_dataset = SQLDataset(train_names, train_queries, train_attrs, train_labels, tokenizer)
    val_dataset = SQLDataset(val_names, val_queries, val_attrs, val_labels, tokenizer)

    # Create dataloaders
    logging.info('Creating dataloaders...')
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    # Train the model
    logging.info('Training model...')
    model = train_model(model, train_dataloader, val_dataloader)

    # Make predictions
    logging.info('Making predictions...')
    predictions = predict(model, val_dataloader, tokenizer)

    # Print some example predictions
    for true, pred in zip(val_labels[:5], predictions[:5]):
        logging.info(f"True: {true}")
        logging.info(f"Predicted: {pred}")

if __name__ == '__main__':
    main()
