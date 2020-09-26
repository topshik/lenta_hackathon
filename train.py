from encodings import hz

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from model import TamerNet3000


def train():
    N_ARTICLES = 105609

    wandb_logger = WandbLogger(offline=False)

    checkpoint_callback = ModelCheckpoint(
        filepath="/".join(["checkpoints", "{epoch}-{val_loss:.2f}-{metric:.2f}"]),
        save_top_k=3,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    model = TamerNet3000(n_articles=N_ARTICLES)
    trainer = pl.Trainer(
        max_epochs=10,
        logger=wandb_logger,
        checkpoint_callback=checkpoint_callback
    )

    trainer.fit(model)


if __name__ == "__main__":
    train()
