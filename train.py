from encodings import hz

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from model import TamerNet3000


wandb_logger = WandbLogger(offline=False)
model = TamerNet3000(n_articles=hz)
trainer = pl.Trainer(
    max_epochs=10,
    gpus=1,
    auto_select_gpus=True,
    logger=wandb_logger
)

trainer.train()