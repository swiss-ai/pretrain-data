from datatrove.pipeline.base import PipelineStep


class XLMRobertaEmbeddingAnnotator(PipelineStep):
    name = "XLM-RoBERTa"
    type = "📄 EMBEDDINGS"

    def __init__(
        self,
        tokenizer_batch_size=10000,
        model_batch_size: int = 2048,
        model: str = "FacebookAI/xlm-roberta-base",
    ):
        super().__init__()
        self.model = model
        self.tokenizer_batch_size = tokenizer_batch_size
        self.model_batch_size = model_batch_size

    def mean_pooling(self, model_output, attention_mask):
        import torch

        # https://huggingface.co/aditeyabaral/sentencetransformer-xlm-roberta-base
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def run(self, data, rank: int = 0, world_size: int = 1):
        from datatrove.utils.batching import batched
        from transformers import AutoTokenizer, AutoModel
        import torch
        import os

        # device = f"cuda:{rank % torch.cuda.device_count()}"
        print(os.environ["SLURM_LOCALID"])
        device = f"cuda:{int(os.environ['SLURM_LOCALID']) % torch.cuda.device_count()}"

        tokenizer = AutoTokenizer.from_pretrained(self.model)
        model = AutoModel.from_pretrained(
            self.model, device_map=device, torch_dtype=torch.bfloat16
        )

        print(f"Rank {rank}, device: {model.device}")

        with self.track_time():
            for batch in batched(data, self.tokenizer_batch_size):
                texts = [d.text for d in batch]

                batch_dict = tokenizer(
                    texts,
                    max_length=512,
                    padding=True,
                    truncation=True,
                    return_overflowing_tokens=True,
                    return_tensors="pt",
                ).to(device)
                for idxs in batched(
                    range(batch_dict["input_ids"].shape[0]),
                    self.model_batch_size,
                ):
                    input_ids = batch_dict["input_ids"][idxs, :]
                    attention_mask = batch_dict["attention_mask"][idxs, :]
                    overflow_to_sample_mapping = batch_dict[
                        "overflow_to_sample_mapping"
                    ][idxs]

                    with torch.no_grad():
                        outputs = model(
                            input_ids=input_ids, attention_mask=attention_mask
                        )
                        embeddings = self.mean_pooling(outputs, attention_mask)

                    for document_index in range(
                        overflow_to_sample_mapping[0],
                        overflow_to_sample_mapping[-1] + 1,
                    ):
                        document = batch[document_index]
                        if "embeddings" in document.metadata:
                            document.metadata["embeddings"] += embeddings[
                                overflow_to_sample_mapping == document_index, :
                            ].tolist()
                        else:
                            document.metadata["embeddings"] = embeddings[
                                overflow_to_sample_mapping == document_index, :
                            ].tolist()
                yield from batch
