import random
import matplotlib.pyplot as plt
import fasttext
import numpy as np
import os
import datasets
import evaluate
import glob
import pandas as pd
import torch
import pandas as pd
import random
import re
import argparse

from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn, optim
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score

class HFRegressor:
    """
    Multi label classification using MSE loss.
    """
    def __init__(self, model_name, label_columns, num_classes, model_path, checkpoint_name=None):
        self.label_columns = label_columns
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_classes = num_classes
        
        # Load pre-trained model (no classification head)
        self.model = AutoModel.from_pretrained(model_name)

        # Add a regression head on top (we use a simple linear layer)
        self.regression_head = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size), 
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(self.model.config.hidden_size, len(label_columns))
        )
        # Freeze
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.regression_head.parameters():
            param.requires_grad = True

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.regression_head.to(self.device)
        self.model_path = model_path
        self.counter = 0
        os.makedirs(os.path.join(self.model_path, 'plots'), exist_ok=True)

        # Load model and tokenizer
        if checkpoint_name is not None:
            self.regression_head.load_state_dict(torch.load(os.path.join(self.model_path, checkpoint_name)))
            self.regression_head.to(self.device)

    def load_data(self, data_path, train_size, batch_size):
        # Load the dataset using Hugging Face's load_dataset
        dataset = load_dataset("parquet", data_files=data_path)["train"]
        dataset = dataset.shuffle(seed=42)

        # Shuffle and split the dataset (train/validation)
        split_idx = int(len(dataset) * train_size)
        train_dataset = dataset.select(range(split_idx))
        val_dataset = dataset.select(range(split_idx, len(dataset)))

        # Tokenize the text content
        def encode_data(examples):
            encodings = self.tokenizer(
                examples["content"],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            labels = list(zip(*[examples[column] for column in self.label_columns]))

            return {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "labels": torch.tensor(labels, dtype=torch.float),
            }

        # Apply tokenization
        train_dataset = train_dataset.map(encode_data, batched=True)
        val_dataset = val_dataset.map(encode_data, batched=True)

        # Set format for PyTorch DataLoader
        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )
        val_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )

        # Convert datasets to DataLoader for batching
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        return train_dataloader, val_dataloader

    def train(self, data_path, batch_size=64, epochs=10, train_size=0.9):
        # Load data
        train_dataloader, val_dataloader = self.load_data(data_path, train_size, batch_size)

        # Define loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.regression_head.parameters(), lr=2e-4)  # Optimize only regression head

        # Training loop
        for epoch in range(epochs):
            self.model.eval()
            self.regression_head.train()
            running_loss = 0.0

            for batch in tqdm(train_dataloader):
                optimizer.zero_grad()

                # Move data to the correct device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                cls_output = outputs.last_hidden_state[:, 0, :]
                logits = self.regression_head(cls_output)

                # Compute loss
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_dataloader):.4f}")

            # Evaluate after each epoch
            self.test(val_dataloader)

            # Save model after each epoch
            model_save_path = os.path.join(self.model_path, f"hf_regressor_{self.counter}.pth")
            torch.save(self.regression_head.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

        results = self.test(train_dataloader)
        print(f"Train results: {results}")
        results = self.test(val_dataloader)
        print(f"Test results: {results}")

    def test(self, val_dataloader):
        self.model.eval()
        self.regression_head.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                cls_output = outputs.last_hidden_state[:, 0, :]
                logits = self.regression_head(cls_output)

                all_preds.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Calculate MSE loss across all predictions and labels
        mse_loss = np.mean((all_preds - all_labels) ** 2)

        # Calculate accuracy per label and plot confusion matrices
        accuracies = []
        for i in range(len(self.label_columns)):
            local_preds = np.clip(np.round(all_preds[:, i]), 0, self.num_classes - 1)
            local_labels = all_labels[:, i]

            accuracy = accuracy_score(local_labels, local_preds)
            accuracies.append(accuracy)

            cm = confusion_matrix(local_labels, local_preds, normalize='true')
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            fig, ax = plt.subplots()
            disp.plot(ax=ax)
            ax.set_title(f"CM for {self.label_columns[i]} (ACC: {accuracy:.2f})")
            plt.savefig(f"{self.model_path}/plots/{self.label_columns[i]}_{self.counter}.png")
            plt.close()

        avg_accuracy = sum(accuracies) / len(accuracies)
        self.counter += 1

        print(f"Test MSE loss: {mse_loss:.4f}, Average accuracy: {avg_accuracy:.4f}")
        return {"MSE_loss": mse_loss, "Average_accuracy": avg_accuracy}

    def annotate(self, input_data_path, output_data_path, batch_size=1024):
        for file_path in glob.glob(os.path.join(input_data_path, "*.parquet")):
            # Create custom path
            file_name = os.path.basename(file_path)
            output_file_path = os.path.join(output_data_path, file_name)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

            # if os.path.exists(output_file_path):
            #     print(f"Output file '{output_file_path}' already exists. Exiting.")
            #     return

            dataset = load_dataset('parquet', data_files=file_path)
            dataset = dataset["train"]

            # Define a function to tokenize the dataset
            def tokenize_and_predict(batch):
                encodings = self.tokenizer(batch['content'], padding="max_length", truncation=True, return_tensors="pt")
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)

                # Get predictions
                with torch.no_grad():
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    cls_output = outputs.last_hidden_state[:, 0, :]
                    logits = self.regression_head(cls_output)
                logits = np.clip(np.round(logits.cpu().numpy()), 0, self.num_classes-1)

                # Assign predictions for each label column separately
                for i, col in enumerate(self.label_columns):
                    batch[col] = logits[:, i]
                return batch

            # Apply the prediction function in batches
            annotated_dataset = dataset.map(tokenize_and_predict, batched=True, batch_size=batch_size)
            annotated_dataset.to_parquet(output_file_path)
            print(f'SAVING DATASET to :{output_file_path} COLUMNS: {self.label_columns}')
            print(annotated_dataset)