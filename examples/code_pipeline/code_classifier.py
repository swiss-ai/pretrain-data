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

from datasets import load_dataset
from tqdm import tqdm
from torch import nn, optim
from transformers import AutoModel, AutoTokenizer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
)


class FastTextClassifier:
    def __init__(self, model_path):
        self.model_path = model_path

    def train(self, data_path, learning_rate, epochs, wordNgrams, dim):
        model = fasttext.train_supervised(
            input=data_path,
            lr=learning_rate,
            epoch=epochs,
            wordNgrams=wordNgrams,
            dim=dim,
            bucket=2000000,
            minCount=5,
            loss="softmax",
        )
        print(model.labels)
        os.makedirs(self.model_path, exist_ok=True)
        model.save_model(f"{self.model_path}/classifier_fasttext.bin")

    def annotate(self, data_path, column):
        model = fasttext.load_model(f"{self.model_path}/classifier_fasttext.bin")

        def format_to_fasttext(example):
            text = repr(example["content"])
            pred = model.predict(text, k=1)
            pred = pred[0][0].replace("__label__", "")
            example[column] = int(pred[0])
            return example

        dataset = datasets.load_dataset("parquet", data_files=data_path)["train"]
        dataset = dataset.map(format_to_fasttext)
        dataset.to_parquet(data_path)
        print(dataset)
        print(set(dataset[column]))
        return dataset

    def test(self, data_path, plot_name):
        model = fasttext.load_model(f"{self.model_path}/classifier_fasttext.bin")
        true_labels, pred_labels = [], []

        with open(data_path, "r") as f:
            for line in f:
                # Split label and text
                label, text = line.strip().split(" ", 1)
                label = label.replace("__label__", "")
                predicted_label = model.predict(repr(text), k=1)
                predicted_label = predicted_label[0][0].replace("__label__", "")
                true_labels.append(label)
                pred_labels.append(predicted_label[0])

        print(set(true_labels), set(pred_labels))
        # Convert labels to numpy arrays for compatibility
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)

        # Calculate accuracy
        accuracy = accuracy_score(true_labels, pred_labels)
        print(f"Accuracy {data_path}: {accuracy:.4f}")

        # Generate confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots()  # Create a figure and axes
        disp.plot(ax=ax)  # Plot on the created axes
        ax.set_title(f"CM  (ACC: {accuracy:.2f})")
        plt.savefig(f"{self.model_path}/{plot_name}.png")
        plt.close()


class MultiFastTextClassifier:
    def __init__(self, model_path):
        self.model_path = model_path

    def train(self, data_path, learning_rate, epochs, wordNgrams, dim):
        # Training the model with logistic loss for multi-label classification
        model = fasttext.train_supervised(
            input=data_path,
            lr=learning_rate,
            epoch=epochs,
            wordNgrams=wordNgrams,
            dim=dim,
            bucket=2000000,
            minCount=1,
            loss="ova",  # Change loss to 'logistic' for multi-label classification
        )
        os.makedirs(self.model_path, exist_ok=True)
        print(model.labels)
        model.save_model(f"{self.model_path}/classifier_fasttext.bin")

    def annotate(self, data_path, columns):
        model = fasttext.load_model(f"{self.model_path}/classifier_fasttext.bin")

        def format_to_fasttext(example):
            text = repr(example["content"])
            # Predicting multiple labels (thresholding)
            pred = model.predict(text, k=-1)
            # re.findall(r'__label__(\w)', TEXT)
            labels = [
                label.replace("__label__", "") for label in pred[0]
            ]  # Remove '__label__' prefix
            print(labels)
            for i, column in enumerate(columns):
                example[column] = labels[i]
            return example

        dataset = datasets.load_dataset("parquet", data_files=data_path)["train"]
        dataset = dataset.map(format_to_fasttext)
        dataset.to_parquet(data_path)
        print(dataset)
        return dataset

    def test(self, data_path, columns, plot_name):
        model = fasttext.load_model(f"{self.model_path}/classifier_fasttext.bin")
        print(model.labels)
        true_labels, pred_labels = [], []

        with open(data_path, "r") as f:
            for line in f:
                # Split label and text
                extracted = line.strip().split(" ", 4)  # Assuming we have 4 columns
                labels, text = extracted[:4], extracted[-1]
                labels = [label.replace("__label__", "") for label in labels]
                true_labels.append(labels)
                # Predicting multiple labels for each document
                predicted_labels = model.predict(
                    repr(text), k=-1
                )  # Get all predicted labels
                predicted_labels = [
                    label.replace("__label__", "") for label in predicted_labels[0]
                ]
                pred_labels.append(predicted_labels)

        for i, column in enumerate(columns):
            column_true_labels = [label[i] for label in true_labels]
            column_pred_labels = [label[i] for label in pred_labels]

            # Convert to numpy arrays and calculate accuracy and confusion matrix
            true_labels_flat = [
                label for sublist in column_true_labels for label in sublist.split()
            ]
            pred_labels_flat = [
                label for sublist in column_pred_labels for label in sublist
            ]

            print(set(true_labels_flat), set(pred_labels_flat))
            true_labels_flat = np.array(true_labels_flat)
            pred_labels_flat = np.array(pred_labels_flat)

            # Calculate accuracy (using multilabel accuracy here)
            accuracy = accuracy_score(true_labels_flat, pred_labels_flat)
            print(f"Accuracy {data_path} ({column}): {accuracy:.4f}")

            # Generate confusion matrix
            cm = confusion_matrix(true_labels_flat, pred_labels_flat, normalize="true")
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            ax = disp.plot()
            ax.set_title(f"CM for {column} (ACC: {accuracy:.2f})")
            plt.savefig(f"{self.model_path}/{column}_{plot_name}.png")
            plt.close()


def fasttext_train(language, columns):
    for column in columns:
        classifier = FastTextClassifier(
            f"./models/fasttext/{language}/{column}"
        )
        classifier.train(
            f"./quality/{language}/postprocessed/fasttext/{column}/train.txt",
            learning_rate=0.05,
            epochs=10,
            wordNgrams=2,
            dim=400,
        )
        classifier.test(
            f"./quality/{language}/postprocessed/fasttext/{column}/train.txt",
            "cm_train.png",
        )
        classifier.test(
            f"./quality/{language}/postprocessed/fasttext/{column}/test.txt",
            "cm_test.png",
        )


def fasttext_annotate(language, columns):
    for column in columns:
        classifier = FastTextClassifier(
            f"./models/fasttext/{language}/{column}"
        )
        for file in tqdm(
            glob.glob(f"./datasets/{language}/*.parquet")
        ):
            classifier.annotate(file, column)


class HFClassifier:
    def __init__(self, model_name, label_column, num_classes, model_path):
        self.label_column = label_column
        self.num_classes = num_classes
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=self.num_classes
        )
        # Freeze all layers except for the classification head
        for param in self.model.base_model.parameters():
            param.requires_grad = False

        self.device = "cuda:0"
        self.model.to(self.device)
        self.model_path = model_path
        self.counter = 0
        os.makedirs(os.path.join(self.model_path, "plots"), exist_ok=True)

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
            encodings = self.tokenizer(
                examples["content"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            labels = torch.tensor(examples[self.label_column], dtype=torch.long)
            return {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "labels": labels,
            }

        # Apply tokenization and mapping to the datasets
        train_dataset = train_dataset.map(
            encode_data, batched=True, batch_size=batch_size
        )
        val_dataset = val_dataset.map(encode_data, batched=True, batch_size=batch_size)

        return {"train": train_dataset, "validation": val_dataset}

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
            compute_metrics=self.compute_metrics,
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
        cm = confusion_matrix(labels, predictions, normalize="true")

        # Display
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots()
        disp.plot(ax=ax)
        ax.set_title(f"CM for {self.label_column} (ACC: {accuracy:.2f})")
        plt.savefig(f"{self.model_path}/plots/{self.counter}.png")
        plt.close()
        self.counter += 1
        return {"accuracy": accuracy}

    def annotate(self, data_path, batch_size=512):
        # Load model and tokenizer
        self.model.load_state_dict(
            torch.load(os.path.join(self.model_path, "hf_classifier.pth"))
        )
        self.model.to(self.device)
        self.model.eval()
        dataset = load_dataset("parquet", data_files=data_path)
        dataset = dataset["train"].select(range(100))

        # Define a function to tokenize the dataset
        def tokenize_and_predict(batch):
            # Tokenize the content
            encodings = self.tokenizer(
                batch["content"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)
            # Make predictions
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                batch[f"cb_{self.label_column}"] = (
                    torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                )
            return batch

        # Apply the prediction function in batches
        annotated_dataset = dataset.map(
            tokenize_and_predict, batched=True, batch_size=batch_size
        )
        for sample in annotated_dataset:
            print(sample[f"ch{self.label_column}"])


class HFRegressor:
    def __init__(self, model_name, label_column, num_classes, model_path):
        self.label_column = label_column
        self.num_classes = num_classes
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1
        )
        # Freeze all layers except for the classification head
        for param in self.model.base_model.parameters():
            param.requires_grad = False

        self.device = "cuda:0"
        self.model.to(self.device)
        self.model_path = model_path
        self.counter = 0
        os.makedirs(os.path.join(self.model_path, "plots"), exist_ok=True)

    def load_data(self, data_path, train_size, batch_size):
        # Load the dataset using Hugging Face's load_dataset
        dataset = load_dataset("parquet", data_files=data_path)

        # Shuffle and split the dataset (train/validation)
        dataset = dataset["train"]
        split_idx = int(len(dataset) * train_size)
        train_dataset = dataset.select(range(200))
        val_dataset = dataset.select(range(200, 400))

        # Tokenize the text content
        def encode_data(examples):
            encodings = self.tokenizer(
                examples["content"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            labels = torch.tensor(examples[self.label_column], dtype=torch.float)
            return {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "labels": labels,
            }

        # Apply tokenization and mapping to the datasets
        train_dataset = train_dataset.map(
            encode_data, batched=True, batch_size=batch_size
        )
        val_dataset = val_dataset.map(encode_data, batched=True, batch_size=batch_size)

        return {"train": train_dataset, "validation": val_dataset}

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
            compute_metrics=self.compute_metrics,
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
        predictions = torch.tensor(predictions).squeeze()
        print(predictions)
        labels = torch.tensor(labels).squeeze().int()
        print(labels)
        predictions = torch.clamp(predictions.round(), min=0, max=5).int()

        # Metrics
        accuracy = accuracy_score(labels, predictions)
        cm = confusion_matrix(labels, predictions, normalize="true")

        # Display
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots()
        disp.plot(ax=ax)
        ax.set_title(f"CM for {self.label_column} (ACC: {accuracy:.2f})")
        plt.savefig(f"{self.model_path}/plots/{self.counter}.png")
        plt.close()
        self.counter += 1
        return {"accuracy": accuracy}

    def annotate(self, data_path, batch_size=512):
        # Load model and tokenizer
        self.model.load_state_dict(
            torch.load(os.path.join(self.model_path, "hf_classifier.pth"))
        )
        self.model.to(self.device)
        self.model.eval()
        dataset = load_dataset("parquet", data_files=data_path)
        dataset = dataset["train"].select(range(100))

        # Define a function to tokenize the dataset
        def tokenize_and_predict(batch):
            # Tokenize the content
            encodings = self.tokenizer(
                batch["content"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)
            # Make predictions
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                batch[f"cb_{self.label_column}"] = (
                    torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                )
            return batch

        # Apply the prediction function in batches
        annotated_dataset = dataset.map(
            tokenize_and_predict, batched=True, batch_size=batch_size
        )
        for sample in annotated_dataset:
            print(sample[f"ch{self.label_column}"])
