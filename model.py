from itertools import chain
from typing import Iterable, List, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from data import LentaDataset


class TamerNet3000(pl.LightningModule):
    def __init__(self, n_articles, embedding_size=128, hidden_size=128) -> None:
        super().__init__()
        self.n_articles = n_articles

        # model
        self.basket_embed = nn.Embedding(n_articles,
                                         embedding_size,
                                         padding_idx=self.n_articles)
        self.rnn = torch.nn.GRU(hidden_size=hidden_size,
                                batch_first=True)
        self.final_dense = nn.Linear(hidden_size, 1)

        self.loss = BCEWithLogitsLoss()

    def prepare_data(self) -> None:
        transactions = pd.read_parquet("hack_data/transactions.parquet", engine="pyarrow")
        self.train_dataset = LentaDataset(transactions)
        self.val_dataset = LentaDataset(transactions)

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(dataset=self.train_dataset,
                                  batch_size=16,
                                  shuffle=True,
                                  num_workers=8,
                                  pin_memory=torch.cuda.is_available(),
                                  drop_last=True)

        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(dataset=self.val_dataset,
                                batch_size=16,
                                shuffle=True,
                                num_workers=8,
                                pin_memory=torch.cuda.is_available(),
                                drop_last=True)

        return val_loader

    def forward(self, batch: Tuple[List[List[List[int, ...]]], List[List[int, ...]], List[int]]) -> torch.float64:
        """
        :param batch: tuple of three objects:
            - transactions history (list of baskets with goods represented with their ordinal number)
            - list of const features: gender, year of birth etc.
            - binary target for prediction
        :return:
        """
        transaction_hist, user_features, target = batch
        user_features, target = (torch.tensor(user_features, device=self.device),
                                 torch.tensor(target, device=self.device))

        baskets_lens = [[len(bill) for bill in user] for user in transaction_hist]

        # batch_size x (n_time_stamps_i x basket_size_i)
        flattened_transactions = [torch.tensor(list(chain.from_iterable(user))) for user in transaction_hist]

        # batch_size x max_{i}(n_time_stamps_i x basket_size_i)
        flattened_transactions_padded = torch.nn.utils.rnn.pad_sequence(flattened_transactions,
                                                                        batch_first=True,
                                                                        padding_value=self.n_artiles)

        # batch_size x max_{i}(n_time_stamps_i x basket_size_i) x embedding_size
        baskets_embeddings = self.basket_embed(flattened_transactions_padded)

        output, h_n = self.rnn(baskets_embeddings)

        logits = self.final_dense(h_n)

        return logits

    def training_step(self, batch):
        logits = self(batch)
        loss = self.loss(logits)

        return {"loss": loss, "logs": {"Training loss": loss.item()}}

    def validation_step(self, batch):
        logits = self(batch)
        loss = self.loss(logits)

        return {"loss": loss.item(), "log": {"Training loss": loss.item()}}