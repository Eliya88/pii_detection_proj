"""PyTorch Dataset for DeBERTa token classification."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast


class PIIDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        tokenizer: PreTrainedTokenizerFast,
        label2id: dict[str, int],
        max_length: int = 512,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        tokens: list[str] = record["tokens"]
        word_labels: list[str] = record["labels"]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        label_ids: list[int] = []
        prev_word_idx: int | None = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)          # [CLS], [SEP], padding
            elif word_idx != prev_word_idx:
                label = word_labels[word_idx]
                label_ids.append(self.label2id.get(label, self.label2id["O"]))
            else:
                label_ids.append(-100)          # continuation sub-word
            prev_word_idx = word_idx

        item = {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(label_ids, dtype=torch.long),
        }
        # DeBERTa-v3 does not use token_type_ids
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        return item
