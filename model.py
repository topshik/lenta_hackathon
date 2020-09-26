from itertools import chain
from typing import Iterable, List, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from data import collate_fn, LentaDataset


class TamerNet3000(pl.LightningModule):
    def __init__(self, n_articles, embedding_size=32, hidden_size=128) -> None:
        super().__init__()
        self.n_articles = n_articles

        # model
        self.basket_embed = nn.Embedding(n_articles + 1,
                                         embedding_size,
                                         padding_idx=self.n_articles)
        self.rnn = torch.nn.GRU(input_size=embedding_size,
                                hidden_size=hidden_size,
                                batch_first=True)
        self.final_dense = nn.Linear(hidden_size, 1)

        self.loss = BCEWithLogitsLoss()

    def prepare_data(self) -> None:
        transactions = pd.read_parquet("hack_data/transactions_cut.parquet", engine="pyarrow")
        transactions = transactions.sample(frac=0.05)
        self.train_dataset = LentaDataset(transactions, [0, 1, 2])
        self.val_dataset = LentaDataset(transactions, [1, 2, 3])

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(dataset=self.train_dataset,
                                  batch_size=16,
                                  shuffle=True,
                                  num_workers=10,
                                  pin_memory=torch.cuda.is_available(),
                                  drop_last=True,
                                  collate_fn=collate_fn)

        print("TRAINING LOADER CREATED")
        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(dataset=self.val_dataset,
                                batch_size=16,
                                shuffle=False,
                                num_workers=10,
                                pin_memory=torch.cuda.is_available(),
                                drop_last=True,
                                collate_fn=collate_fn)

        print("VALIDATION LOADER CREATED")
        return val_loader

    def forward(self, batch: Tuple[List[List[List[int]]], List[List[int]], List[int]]) -> torch.float64:
        """
        :param batch: tuple of three objects:
            - transactions history (list of baskets with goods represented with their ordinal number)
            - list of const features: gender, year of birth etc.
            - binary target for prediction
        :return:
        """
        flattened_transactions_padded, basket_lens, target = batch
        # client_features = (torch.tensor(user_features, device=self.device))

        # batch_size x max_{i}(n_time_stamps_i x basket_size_i) x embedding_size
        basket_embeddings_flattened = self.basket_embed(flattened_transactions_padded)

        basket_embeddings = []
        for basket_lens_in_chq, embeddings in zip(basket_lens, basket_embeddings_flattened):
            i = 0
            basket_embeddings.append([])
            for basket_len in basket_lens_in_chq:
                basket_embeddings[-1].append(embeddings[i:i + basket_len].mean(0))
                i += basket_len
            basket_embeddings[-1] = torch.stack(basket_embeddings[-1])

        basket_embeddings_padded = torch.nn.utils.rnn.pad_sequence(
            basket_embeddings,
            batch_first=True,
            # padding_value=torch.zeros(basket_embeddings_flattened.shape[-1])
            padding_value=0.0
        )

        output, h_n = self.rnn(basket_embeddings_padded)

        # client_features = torch.cat([h_n, client_features], 1)
        client_features = h_n

        logits = self.final_dense(client_features)

        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.loss(logits.squeeze(), torch.tensor(batch[2], device=self.device, dtype=torch.float))

        return {"loss": loss, "logits": logits, "log": {"Training loss": loss.item()}}

    def training_epoch_end(self, outputs):
        epoch_loss = torch.stack([x["loss"] for x in outputs]).mean()

        # Metrics
        # acc_top1 = torch.stack([torch.tensor(x["acc_top1"]) for x in outputs]).mean()

        logs = {"train/epoch_loss": epoch_loss,
                "train/metric_coooooool": 0.0}

        return {"val_loss": epoch_loss, "metric": 0.0, "log": logs}

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.loss(logits.squeeze(), torch.tensor(batch[2], device=self.device, dtype=torch.float))

        return {"loss": loss, "logits": logits, "log": {"Validation loss": loss.item()}}

    def validation_epoch_end(self, outputs):
        epoch_loss = torch.stack([x["loss"] for x in outputs]).mean()

        # Metrics
        # acc_top1 = torch.stack([torch.tensor(x["acc_top1"]) for x in outputs]).mean()

        logs = {"val/epoch_loss": epoch_loss,
                "val/metric_coooooool": 0.0}

        return {"val_loss": epoch_loss, "metric": 0.0, "log": logs}

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-4)
