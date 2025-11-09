import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List


class BertEncoder:
    """DistilBERT sentence embeddings via mean pooling."""

    def __init__(self, model_name: str = "distilbert-base-uncased", device: str | None = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 32, max_length: int = 128) -> np.ndarray:
        embs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            toks = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**toks).last_hidden_state
            mask = toks["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            summed = (hidden * mask).sum(1)
            counts = mask.sum(1).clamp(min=1e-9)
            embs.append((summed / counts).cpu().numpy())
        return np.vstack(embs)
