import pandas as pd
import torch
import torch.utils.data as torchdata


class LentaDataset(torchdata.Dataset):
    """
    Dataset with client ids, chqs and materials
    :param transactions: initial transactions table
    """
    def __init__(self, transactions: pd.DataFrame) -> None:
        self.clients_to_chqs = transactions.groupby(["client_id"])["chq_id"].apply(list).to_frame().reset_index()
        self.clients_to_chqs = self.clients_to_chqs.set_index("client_id")
        self.chqs_to_materials = transactions.groupby(["chq_id"])["material_encoded"].apply(
            list).to_frame().reset_index()
        self.chqs_to_materials = self.chqs_to_materials.set_index("chq_id")

    def __len__(self):
        return len(self.clients_to_chqs)

    def __getitem__(self, idx):
        data = [
            self.chqs_to_materials.loc[chq]["material_encoded"]
            for chq in self.clients_to_chqs.iloc[0]["chq_id"]
        ]

        return data
