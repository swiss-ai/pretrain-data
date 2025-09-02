import torch
import torch.nn as nn
import torch.optim as optim
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer

model_ft = AutoModel.from_pretrained("FacebookAI/xlm-roberta-base")
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")


def tokenize(batch):
    return tokenizer(batch["content"], padding=True, truncation=True, max_length=512)


# Define the model with an MLP classifier on top of RoBERTa
class RobertaClassifier(nn.Module):
    def __init__(
        self, num_classes, model_name="FacebookAI/xlm-roberta-base", device="cuda:0"
    ):
        super(RobertaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.freeze_roberta_encoder()
        self.device = device
        self.classifier = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, self.roberta.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.roberta.config.hidden_size, num_classes),
        )

    def freeze_roberta_encoder(self):
        for param in self.roberta.parameters():
            param.requires_grad = False

    def mean_pooling(self, model_output, attention_mask):
        import torch

        # https://huggingface.co/aditeyabaral/sentencetransformer-xlm-roberta-base
        token_embeddings = (
            model_output.last_hidden_state
        )  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, input_ids=None, attention_mask=None, roberta_embeddings=None):
        # outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # pooled_output = outputs.last_hidden_state[:, 0]  # CLS token representation
        if roberta_embeddings is None:
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            roberta_embeddings = self.mean_pooling(outputs, attention_mask)
        logits = self.classifier(roberta_embeddings)
        return torch.nn.functional.softmax(logits, dim=1)

    def predict(self, input_ids=None, attention_mask=None, roberta_embeddings=None):
        """
        Predicts class labels for a list of texts.

        Args:
            texts (list of str): The input sentences to classify.
            max_length (int): Maximum sequence length for tokenization.

        Returns:
            list of int: Predicted class labels for each input text.
        """
        self.eval()

        with torch.no_grad():
            if roberta_embeddings is None:
                logits = self(input_ids, attention_mask)
            else:
                logits = self(roberta_embeddings=roberta_embeddings)
        return logits[:, 1].cpu().numpy()


def train_model(
    model,
    train_dataset,
    test_dataset,
    num_epochs=6,
    learning_rate=3e-4,
    batch_size=64,
    model_save_path="./swiss-ai/detoxify_mlp_multilingual/best_model.pth",
):
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    eval_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Hyperparameters and configurations
    num_classes = 2  # Update based on your dataset
    # Initialize model, optimizer, and scheduler
    model = RobertaClassifier(num_classes)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=200, num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    # Training and validation loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}")

        # Validation
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == labels)

        avg_val_loss = total_val_loss / len(eval_loader)
        val_accuracy = correct_predictions.double() / len(test_dataset)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy:.4f}"
        )

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with Validation Loss: {avg_val_loss:.4f}")

    print("Training complete.")


import argparse

args_parser = argparse.ArgumentParser()
args_parser.add_argument(
    "--processed_dir",
    default="./multilingual_pretrain/detoxify_models",
    type=str,
)
args_parser.add_argument(
    "--model_save_dir",
    default="./swiss-ai/detoxify_mlp_multilingual",
    type=str,
)
if __name__ == "__main__":
    args = args_parser.parse_args()
    PROCESSED_DIR = args.processed_dir
    LANG2ProcessPATH = {
        "dutch": f"{PROCESSED_DIR}/Dutch_processed.csv",
        "french": f"{PROCESSED_DIR}/French_processed.csv",
        "german": f"{PROCESSED_DIR}/German_processed.csv",
        "italian": f"{PROCESSED_DIR}/Italian_processed.csv",
        "polish": f"{PROCESSED_DIR}/Polish_processed.csv",
        "spanish": f"{PROCESSED_DIR}/Spanish_processed.csv",
        "portuguese": f"{PROCESSED_DIR}/Portuguese_processed.csv",
        "english": f"{PROCESSED_DIR}/English_processed.csv",
    }

    for lang, datapath in LANG2ProcessPATH.items():
        print(f"================= {lang} =================")
        dataset = load_dataset("csv", data_files=datapath)
        dataset = dataset["train"].train_test_split(test_size=0.1)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        train_dataset = train_dataset.map(
            tokenize, batched=True, batch_size=len(train_dataset)
        )
        test_dataset = test_dataset.map(
            tokenize, batched=True, batch_size=len(test_dataset)
        )
        train_dataset = train_dataset.remove_columns(["content"])
        test_dataset = test_dataset.remove_columns(["content"])
        train_dataset = train_dataset.rename_columns({"toxic": "label"})
        test_dataset = test_dataset.rename_columns({"toxic": "label"})
        train_dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "label"]
        )
        test_dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "label"]
        )
        print(f"Training Toxicity Classifier on {lang}...")
        train_model(
            model_ft,
            train_dataset,
            test_dataset,
            model_save_path=f"{args.model_save_dir}/{lang}.pth",
        )
        print("========================================")
