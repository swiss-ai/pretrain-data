from data_pipeline_pretrain.utils import list_files
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
import torch
import os
from tqdm.contrib.concurrent import thread_map
from functools import partial
from pyarrow.parquet import ParquetFile
import pyarrow as pa
import numpy as np


class EmbeddingBinaryClassifierFilter(BaseFilter):
    name = "BINARY CLASSIFIER"
    type = "🖩 EMBEDDINGS FILTER"

    def __init__(
        self,
        classifier,
        threshold: float,
        batch_size: int = 10_000,
        embedding_key=lambda x: x.metadata["embeddings"][0],
        exclusion_writer: DiskWriter = None,
    ):
        super().__init__(batch_size=batch_size, exclusion_writer=exclusion_writer)
        self.classifier = classifier
        self.embedding_key = embedding_key
        self.threshold = threshold

    def filter(self, document):
        pass

    def filter_batch(self, batch):
        import torch

        embeddings = torch.tensor([self.embedding_key(document) for document in batch])
        with torch.no_grad():
            scores = torch.nn.functional.sigmoid(self.classifier(embeddings)).flatten()
        for document, score in zip(batch, scores):
            document.metadata["quality_score"] = score.item()
        return map(lambda x: x > self.threshold, scores)


class BinaryClassifier(torch.nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=256):
        super(BinaryClassifier, self).__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, X):
        return self.classifier(X)

    def to_pt(self, file_name):
        file_name = os.path.normpath(file_name)
        torch.save(self.state_dict(), file_name)

    def from_pt(file_name, embedding_dim=768, hidden_dim=256):
        file_name = os.path.normpath(file_name)
        state_dict = torch.load(
            file_name, weights_only=True, map_location=torch.device("cpu")
        )
        classifier = BinaryClassifier(
            embedding_dim=embedding_dim, hidden_dim=hidden_dim
        )
        classifier.load_state_dict(state_dict)
        classifier.eval()
        return classifier


def estimate_classifier_threshold(
    input_dir,
    classifier,
    num_samples,
    top_p,
    embedding_key=lambda x: x["embeddings"][0],
    num_workers=16,
):
    files = list_files(input_dir)
    num_samples_per_file = num_samples // len(files)

    def estimate_score(file, num_samples_per_file):
        pf = ParquetFile(file)
        pf = next(
            pf.iter_batches(batch_size=num_samples_per_file, columns=["metadata"])
        )
        df = pa.Table.from_batches([pf]).to_pandas()
        embeddings = torch.tensor(
            np.array(
                [embd for embd in df["metadata"].apply(embedding_key)],
                dtype=np.float32,
            )
        )
        with torch.no_grad():
            classifier_scores = torch.nn.functional.sigmoid(classifier(embeddings))
        return classifier_scores.flatten().tolist()

    scores = thread_map(
        partial(estimate_score, num_samples_per_file=num_samples_per_file),
        files,
        max_workers=num_workers,
    )
    scores = [el for arr in scores for el in arr]

    return np.quantile(scores, 1 - top_p)
