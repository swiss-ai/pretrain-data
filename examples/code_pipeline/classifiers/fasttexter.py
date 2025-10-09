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


class FastTextClassifier:
    def __init__(self, model_path, column):
        self.model_path = model_path
        self.column = column

    def train(self, data_path, learning_rate=0.05, epochs=5, wordNgrams=2, dim=300):
        model = fasttext.train_supervised(
            input=data_path,
            lr=learning_rate,
            epoch=epochs,
            wordNgrams=wordNgrams,
            dim=dim,
            bucket=2000000,
            minCount=5,
            loss='softmax'
        )
        print("Model has: ", len(model.labels), " number of labels.")
        os.makedirs(self.model_path, exist_ok=True)
        model.save_model(f"{self.model_path}/classifier_fasttext.bin")

    def annotate(self, input_data_path, output_data_path):
        model = fasttext.load_model(f"{self.model_path}/classifier_fasttext.bin")

        def format_to_fasttext(example):
            text = repr(example['content'])
            pred = model.predict(text, k=1)
            pred = pred[0][0].replace('__label__', '')
            example[self.column] = int(pred[0])
            return example

        for file_path in glob.glob(os.path.join(input_data_path, "*.parquet")):
            # Create custom path
            file_name = os.path.basename(file_path)
            output_file_path = os.path.join(output_data_path, self.label_column, file_name)
            os.makedirs(output_file_path, exist_ok=True)

            dataset = datasets.load_dataset('parquet', data_files=file_path)['train']
            dataset = dataset.map(format_to_fasttext)
            dataset.to_parquet(output_data_path)
            print(set(dataset[self.column]))


    def test(self, data_path, plot_name):
        model = fasttext.load_model(f"{self.model_path}/classifier_fasttext.bin")
        true_labels, pred_labels = [], []

        with open(data_path, 'r') as f:
            for line in f:
                # Split label and text
                label, text = line.strip().split(' ', 1)
                label = label.replace('__label__', '')
                predicted_label = model.predict(repr(text), k=1)
                predicted_label = predicted_label[0][0].replace('__label__', '')
                true_labels.append(label)
                pred_labels.append(predicted_label[0])

        print("True labels: ", set(true_labels), ", Prediction labels: ", set(pred_labels))
        # Convert labels to numpy arrays for compatibility
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)

        # Calculate accuracy
        accuracy = accuracy_score(true_labels, pred_labels)
        print(f'Accuracy {data_path}: {accuracy:.4f}')

        # Generate confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, normalize='true')
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
            loss='ova'  # Change loss to 'logistic' for multi-label classification
        )
        os.makedirs(self.model_path, exist_ok=True)
        print(model.labels)
        model.save_model(f"{self.model_path}/classifier_fasttext.bin")

    def annotate(self, data_path, columns):
        model = fasttext.load_model(f"{self.model_path}/classifier_fasttext.bin")

        def format_to_fasttext(example):
            text = repr(example['content'])
            # Predicting multiple labels (thresholding)
            pred = model.predict(text, k=-1) 
            #re.findall(r'__label__(\w)', TEXT)
            labels = [label.replace('__label__', '') for label in pred[0]]  # Remove '__label__' prefix
            print(labels)
            for i, column in enumerate(columns): example[column] = labels[i]
            return example

        dataset = datasets.load_dataset('parquet', data_files=data_path)['train']
        dataset = dataset.map(format_to_fasttext)
        dataset.to_parquet(data_path)
        print(dataset)
        return dataset

    def test(self, data_path, columns, plot_name):
        model = fasttext.load_model(f"{self.model_path}/classifier_fasttext.bin")
        print(model.labels)
        true_labels, pred_labels = [], []

        with open(data_path, 'r') as f:
            for line in f:
                # Split label and text
                extracted = line.strip().split(' ', 4) # Assuming we have 4 columns
                labels, text = extracted[:4], extracted[-1]
                labels = [label.replace('__label__', '') for label in labels]
                true_labels.append(labels)
                # Predicting multiple labels for each document
                predicted_labels = model.predict(repr(text), k=-1)  # Get all predicted labels
                predicted_labels = [label.replace('__label__', '') for label in predicted_labels[0]]
                pred_labels.append(predicted_labels)

        for i, column in enumerate(columns):
            column_true_labels = [label[i] for label in true_labels]
            column_pred_labels = [label[i] for label in pred_labels]

            # Convert to numpy arrays and calculate accuracy and confusion matrix
            true_labels_flat = [label for sublist in column_true_labels for label in sublist.split()]
            pred_labels_flat = [label for sublist in column_pred_labels for label in sublist]

            print(set(true_labels_flat), set(pred_labels_flat))
            true_labels_flat = np.array(true_labels_flat)
            pred_labels_flat = np.array(pred_labels_flat)

            # Calculate accuracy (using multilabel accuracy here)
            accuracy = accuracy_score(true_labels_flat, pred_labels_flat)
            print(f'Accuracy {data_path} ({column}): {accuracy:.4f}')

            # Generate confusion matrix
            cm = confusion_matrix(true_labels_flat, pred_labels_flat, normalize='true')
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            ax = disp.plot()
            ax.set_title(f"CM for {column} (ACC: {accuracy:.2f})")
            plt.savefig(f"{self.model_path}/{column}_{plot_name}.png")
            plt.close()
