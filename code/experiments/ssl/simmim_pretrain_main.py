import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI

from models import ViTSimMIM
from torch.nn import L1Loss
from monai.inferers import SlidingWindowInferer
from utils.schedulers import LinearWarmupCosineAnnealingLR
import data
import optimizers


class SimMIMtrainer(pl.LightningModule):
    """Pretraining on 3D Imaging with Masked Auto Encoder"""

    def __init__(
        self, model_name: str, model_dict: dict,
    ):
        super().__init__()
        self.model_name = model_name
        self.model_dict = model_dict

        self.model = ViTSimMIM(**model_dict)

        self.recon_loss = L1Loss()
        self.recon_patches = []
        # self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # --------------------------
        image = batch["image"]
        pred_pixel_values, patches, batch_range, masked_indices = self.model(image)
        batch_size = pred_pixel_values.shape[0]
        loss = self.recon_loss(pred_pixel_values, patches[batch_range, masked_indices])

        self.log("train/l1_loss", loss, batch_size=batch_size, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # --------------------------
        image = batch["image"]
        pred_pixel_values, patches, batch_range, masked_indices = self.model(image)
        batch_size = pred_pixel_values.shape[0]
        loss = self.recon_loss(pred_pixel_values, patches[batch_range, masked_indices])

        self.log("val/l1_loss", loss, batch_size=batch_size, sync_dist=True)

        return {"val_loss": loss, "val_number": batch_size}

    def validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_loss = torch.tensor(val_loss / len(outputs))

        self.log(
            "val/l1_loss_avg", mean_val_loss, sync_dist=True,
        )
        self.logger.log_hyperparams(
            params={
                "model": self.model_name,
                **self.model_dict,
                # "data": self.trainer.datamodule.json_path,
                # "ds_ratio": self.trainer.datamodule.downsample_ratio,
                "batch_size": self.trainer.datamodule.batch_size,
                "distribution": self.trainer.datamodule.dist,
                # "benchmark": self.trainer.benchmark,
                "max_epochs": self.trainer.max_epochs,
                "precision": self.trainer.precision,
            },
            metrics={"l1_loss": mean_val_loss},
        )


if __name__ == "__main__":
    cli = LightningCLI(save_config_overwrite=True)
