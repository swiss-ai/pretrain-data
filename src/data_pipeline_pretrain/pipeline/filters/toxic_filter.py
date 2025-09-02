from datatrove.pipeline.base import PipelineStep
import torch
import torch.nn as nn
from datatrove.pipeline.writers.disk_base import DiskWriter
from data_pipeline_pretrain.utils import list_files
from transformers import RobertaModel
from pyarrow.parquet import ParquetFile
import pyarrow as pa


class RobertaClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        model_name="FacebookAI/xlm-roberta-base",
        device="cuda:0",
        cls_only=False,
    ):
        super(RobertaClassifier, self).__init__()
        self.cls_only = cls_only
        if not cls_only:
            self.roberta = RobertaModel.from_pretrained(model_name)
            self.freeze_roberta_encoder()
        self.device = device
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(768, num_classes),
        )

    def freeze_roberta_encoder(self):
        for param in self.roberta.parameters():
            param.requires_grad = False

    def mean_pooling(self, model_output, attention_mask):
        # https://huggingface.co/aditeyabaral/sentencetransformer-xlm-roberta-base
        import torch

        token_embeddings = (
            model_output.last_hidden_state
        )  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        roberta_embeddings=None,
        dtype=torch.float,
    ):
        import torch

        if not self.cls_only:
            self.roberta = self.roberta.to(dtype)
        if roberta_embeddings is None:
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            roberta_embeddings = self.mean_pooling(outputs, attention_mask)
        roberta_embeddings = roberta_embeddings.to(dtype)
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
        import torch

        self.eval()

        with torch.no_grad():
            if roberta_embeddings is None:
                assert not self.cls_only
                logits = self(input_ids, attention_mask)
            else:
                logits = self(roberta_embeddings=roberta_embeddings)
        return logits[:, 1].cpu().numpy()


class ToxicScorer(PipelineStep):
    name = "ToxicScorer"
    type = "Content Filtering"

    def __init__(
        self,
        model_path: str = "./multilingual_pretrain/detoxify_models/toxic_model.pth",
        batch_size: int = 2048,
    ):
        super().__init__()
        self.model_path = model_path
        self.batch_size = batch_size

    def load_model(self, dtype=torch.float):
        # Load the model and set it to evaluation mode
        self.model = RobertaClassifier(num_classes=2, cls_only=True)
        # only load the classifier instead of the entire model
        self.model.classifier.load_state_dict(torch.load(self.model_path))
        self.model = self.model.to(dtype)

    def run(self, data, rank: int = 0, world_size: int = 1, verbose: bool = False):
        from datatrove.utils.batching import batched
        import torch
        import os
        import numpy as np

        self.load_model()
        device = "cpu"
        model = self.model.to(device)

        with self.track_time():
            for batch in batched(data, self.batch_size):
                embed_batch = [
                    torch.tensor(d.metadata["embeddings"], dtype=torch.float).to(device)
                    for d in batch
                ]
                doc_ids_batch = [
                    torch.ones(len(d.metadata["embeddings"])) * idx
                    for idx, d in enumerate(batch)
                ]
                logits = model.predict(
                    roberta_embeddings=torch.cat(embed_batch, dim=0).to(device)
                )
                doc_ids_batch = torch.cat(doc_ids_batch, dim=0)

                if verbose:
                    for embed in embed_batch:
                        print(f"{embed.shape}")

                toxic_scores = []
                for idx in range(len(batch)):
                    logit_batch = logits[(doc_ids_batch == idx).numpy()]
                    score = np.max(logit_batch)
                    toxic_scores.append(score)

                for document, score in zip(batch, toxic_scores):
                    document.metadata["toxic_score"] = score
                    yield document


################################################################
from datatrove.pipeline.filters.base_filter import BaseFilter
import torch


class ToxicityBinaryClassifierFilter(BaseFilter):
    name = "BinaryFilter"
    type = "ToxicityFilter"

    def __init__(
        self,
        threshold: float,
        batch_size: int = 10_000,
        score_key=lambda x: x.metadata["toxic_score"],
        exclusion_writer: DiskWriter = None,
    ):
        super().__init__(batch_size=batch_size, exclusion_writer=exclusion_writer)
        self.score_key = score_key
        self.threshold = threshold

    def filter(self, document):
        pass

    def filter_batch(self, batch):
        import torch

        toxic_scores = torch.tensor([self.score_key(document) for document in batch])
        return map(lambda x: x < self.threshold, toxic_scores)
