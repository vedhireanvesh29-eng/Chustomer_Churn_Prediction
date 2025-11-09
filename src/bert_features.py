import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

class BertEncoder:
    """
    Lightweight sentence embedding using a BERT-family model (no fine-tuning).
    Mean-pools last hidden state. Works offline after first download.
    """
    def __init__(self, model_name="distilbert-base-uncased", device=None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def encode(self, texts, batch_size=32, max_length=128):
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            toks = self.tokenizer(
                batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            ).to(self.device)
            out = self.model(**toks).last_hidden_state   # [B, L, H]
            mask = toks['attention_mask'].unsqueeze(-1).expand(out.size()).float()
            masked = out * mask
            summed = masked.sum(1)
            counts = mask.sum(1).clamp(min=1e-9)
            mean_pooled = summed / counts
            embs.append(mean_pooled.cpu().numpy())
        return np.vstack(embs)
