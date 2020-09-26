from encodings import hz

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from model import TamerNet3000


def train():
    N_ARTICLES = 105609

    # wandb_logger = WandbLogger(offline=False)
    model = TamerNet3000(n_articles=N_ARTICLES)
    trainer = pl.Trainer(
        max_epochs=10,
    )

    trainer.fit(model)


if __name__ == "__main__":
    train()
