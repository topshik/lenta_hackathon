from itertools import chain
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.utils.data as torchdata

import sklearn.preprocessing as skprep


def collate_fn(batch):
    n_articles = 105609

    transaction_hist = [elem[0] for elem in batch]
    target = [elem[1] for elem in batch]
    basket_lens = [[len(bill) for bill in user] for user in transaction_hist]

    # batch_size x (n_time_stamps_i x basket_size_i)
    flattened_transactions = [torch.tensor(list(chain.from_iterable(user))) for user in transaction_hist]

    # batch_size x max_{i}(n_time_stamps_i x basket_size_i)
    flattened_transactions_padded = torch.nn.utils.rnn.pad_sequence(flattened_transactions,
                                                                    batch_first=True,
                                                                    padding_value=n_articles)

    return flattened_transactions_padded, basket_lens, target


class LentaDataset(torchdata.Dataset):
    """
    Dataset with client ids, chqs and materials
    Warning: this class changes "transactions" dataframe
    :param transactions: initaltransactions table
    :param months: list of numbers of months to train on (target will come from the month after the last one)
    month indexing starts from zero
    """

    def __init__(self, transactions: pd.DataFrame, months: List[int]) -> None:
        ### --- Prepare target --- ###
        month_encoder = skprep.LabelEncoder()

        # Make month index from year and month
        transactions["chq_datetime_idx"] = pd.DatetimeIndex(transactions["chq_date"])
        transactions["year_month"] = transactions["chq_datetime_idx"].apply(lambda x: x.strftime("%Y-%m"))
        month_encoder.fit(sorted(transactions["year_month"].unique()))
        transactions["month_ix"] = month_encoder.transform(transactions["year_month"])

        # Calculate client spendings during months in train and test periods
        clients_spendings = transactions.groupby(["client_id", "month_ix"])["sales_sum"].sum().to_frame().reset_index()
        train_spendings = clients_spendings[clients_spendings["month_ix"].isin(months)]
        test_spendings = clients_spendings[clients_spendings["month_ix"] == max(*months) + 1]

        # Make a table of spendings with rows for each user and columns for each month
        # (drop users with no spendings during any month in train period)
        train_spendings_by_month = pd.pivot_table(
            train_spendings, index="client_id", values="sales_sum", columns="month_ix", aggfunc=np.sum
        ).dropna().reset_index()
        # Get minimal spending per month in train period
        train_spendings_by_month["min_train_sum"] = train_spendings_by_month[months].min(axis=1)
        # Join minimal spending in train period with spending in test period
        joined = train_spendings_by_month.join(test_spendings.set_index("client_id"), on="client_id", rsuffix="_right",
                                               how="left")
        # Calculate that a client left if he bought two times less than in previous months
        joined["churn"] = joined["sales_sum"].fillna(0) * 2 < joined["min_train_sum"]
        # Get set of churm clients
        self.churn_clients = set(joined["client_id"][joined["churn"]])

        ### --- Prepare data --- ###
        # Label encode all items
        transactions = transactions[transactions["month_ix"].isin(months)]
        transactions["material_encoded"] = skprep.LabelEncoder().fit_transform(transactions["material"])
        # Create dataframe of clients and their cheqs
        self.clients_to_chqs = transactions.groupby(["client_id"])["chq_id"].apply(list).to_frame().reset_index()
        self.clients_to_chqs = self.clients_to_chqs.set_index("client_id")
        # Create of cheqs and their goods
        self.chqs_to_materials = transactions.groupby(["chq_id"])["material_encoded"].apply(
            list).to_frame().reset_index()
        self.chqs_to_materials = self.chqs_to_materials.set_index("chq_id")

    def __len__(self):
        return len(self.clients_to_chqs)

    def __getitem__(self, idx):
        purchases = [
            torch.tensor(self.chqs_to_materials.loc[chq]["material_encoded"])
            for chq in self.clients_to_chqs.iloc[idx]["chq_id"]
        ]
        target = int(self.clients_to_chqs.index[idx] in self.churn_clients)

        return purchases, target
