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

class HFClassifier:
    def __init__(self, model_name, label_column, num_classes, model_path, checkpoint_name=None):
        self.label_column = label_column
        self.num_classes = num_classes
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=self.num_classes
        )
        # Freeze all layers except for the classification head
        for param in self.model.base_model.parameters():
            param.requires_grad = False

        self.device = "cuda:0"
        self.model.to(self.device)
        self.model_path = model_path
        self.counter = 0
        os.makedirs(os.path.join(self.model_path, 'plots'), exist_ok=True)

        # Load model and tokenizer
        if checkpoint_name is not None:
            self.model.load_state_dict(torch.load(os.path.join(self.model_path, checkpoint_name)))
            self.model.to(self.device)

    def load_data(self, data_path, train_size, batch_size):
        # Load the dataset using Hugging Face's load_dataset
        dataset = load_dataset("parquet", data_files=data_path)

        # Shuffle and split the dataset (train/validation)
        dataset = dataset["train"]
        split_idx = int(len(dataset) * train_size)
        train_dataset = dataset.select(range(split_idx))
        val_dataset = dataset.select(range(split_idx, len(dataset)))

        # Tokenize the text content
        def encode_data(examples):
            encodings = self.tokenizer(examples["content"], padding="max_length", truncation=True, return_tensors="pt")
            labels = torch.tensor(examples[self.label_column], dtype=torch.long)
            return {"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"], "labels": labels}

        # Apply tokenization and mapping to the datasets
        train_dataset = train_dataset.map(encode_data, batched=True, batch_size=batch_size)
        val_dataset = val_dataset.map(encode_data, batched=True, batch_size=batch_size)

        return {
            "train": train_dataset,
            "validation": val_dataset
        }

    def train(self, data_path, batch_size=64, epochs=10, train_size=0.9):
        dataset = self.load_data(data_path, train_size, batch_size)

        # Define the training arguments
        training_args = TrainingArguments(
            output_dir=self.model_path,
            eval_strategy="epoch", 
            save_strategy="epoch",
            logging_strategy="epoch",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.005,
            report_to="wandb",
            learning_rate=2e-5,
            run_name=f"{self.model_path}/{self.label_column}",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        os.makedirs(self.model_path, exist_ok=True)
        model_save_path = os.path.join(self.model_path, "hf_classifier.pth")
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

        results = trainer.evaluate(dataset["validation"])
        print(f"Test results: {results}")
        results = trainer.evaluate(dataset["train"])
        print(f"Train results: {results}")

    def compute_metrics(self, p):
        # Initialize the metric
        predictions, labels = p
        predictions = torch.argmax(torch.tensor(predictions), dim=-1)

        # Metrics
        accuracy = accuracy_score(labels, predictions)
        cm = confusion_matrix(labels, predictions, normalize='true')

        # Display
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots()
        disp.plot(ax=ax)
        ax.set_title(f"CM for {self.label_column} (ACC: {accuracy:.2f})")
        plt.savefig(f"{self.model_path}/plots/{self.counter}.png")
        plt.close()
        self.counter += 1
        return {"accuracy": accuracy}

    def annotate(self, input_data_path, output_data_path, batch_size=512):
        self.model.eval()
        for file_path in glob.glob(os.path.join(input_data_path, "*.parquet")):
            # Create custom path
            file_name = os.path.basename(file_path)
            output_file_path = os.path.join(output_data_path, self.label_column, file_name)
            os.makedirs(output_file_path, exist_ok=True)

            dataset = load_dataset('parquet', data_files=file_path)
            dataset = dataset['train']

            # Define a function to tokenize the dataset
            def tokenize_and_predict(batch):
                # Tokenize the content
                encodings = self.tokenizer(batch['content'], padding="max_length", truncation=True, return_tensors="pt")
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                # Make predictions
                with torch.no_grad():
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    batch[f'{self.label_column}'] = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                return batch

            # Apply the prediction function in batches
            annotated_dataset = dataset.map(tokenize_and_predict, batched=True, batch_size=batch_size)
            annotated_dataset.to_parquet(output_file_path)
            print(f'SAVING DATASET to :{output_file_path} COLUMN: {self.label_column}')
            print(annotated_dataset)