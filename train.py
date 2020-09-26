from encodings import hz

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from model import TamerNet3000


N_ARTICLES = 105609

wandb_logger = WandbLogger(offline=False)
model = TamerNet3000(n_articles=N_ARTICLES)
trainer = pl.Trainer(
    max_epochs=10,
    auto_select_gpus=True,
    logger=wandb_logger
)

trainer.fit(model)
