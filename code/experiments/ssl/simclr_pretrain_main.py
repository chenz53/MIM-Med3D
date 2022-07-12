import data
import optimizers
from models import ViTAutoEnc
from losses import ContrastiveLoss
from torch.nn import L1Loss

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI


class SimCLRtrainer(pl.LightningModule):
    def __init__(
        self, batch_size: int, temperature: float, model_name: str, model_dict: dict
    ):
        super().__init__()
        self.model_name = model_name
        self.model_dict = model_dict
        self.temperature = temperature

        if model_name.split("_")[0] == "vitautoenc_base":
            self.model = ViTAutoEnc(**model_dict)

        # self.recon_loss = L1Loss()
        self.contrastive_loss = ContrastiveLoss(
            batch_size=batch_size * 2, temperature=temperature
        )

        self.log_kwargs = {
            "on_epoch": True,
            # "sync_dist": True,
            "on_step": True,
            "prog_bar": True,
            "logger": True,
        }

        # self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # --------------------------
        # REPLACE WITH YOUR OWN
        inputs, inputs_2 = batch["image"], batch["image_2"]
        batch_size = inputs.shape[0]
        outputs_v1, hidden_v1 = self.model(inputs)
        outputs_v2, hidden_v2 = self.model(inputs_2)

        flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=4)
        flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=4)

        cl_loss = self.contrastive_loss(flat_out_v1, flat_out_v2)

        # Adjust the CL loss by Recon Loss
        # total_loss = r_loss + cl_loss * r_loss

        # self.log("train_loss/recon_loss", r_loss, batch_size=batch_size)
        self.log("train_loss", cl_loss, batch_size=batch_size, **log_kwargs)
        # self.log("train_loss/total_loss", total_loss, batch_size=batch_size)

        return cl_loss
        # --------------------------

    def validation_step(self, batch, batch_idx):
        # --------------------------
        # REPLACE WITH YOUR OWN
        inputs, inputs_2 = batch["image"], batch["image_2"]
        batch_size = inputs.shape[0]
        outputs_v1, hidden_v1 = self.model(inputs)
        outputs_v2, hidden_v2 = self.model(inputs_2)

        flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=4)
        flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=4)

        val_loss = self.contrastive_loss(flat_out_v1, flat_out_v2)

        self.log("val_loss", val_loss, batch_size=batch_size, **log_kwargs)

        return {"val_loss": val_loss, "val_number": batch_size}

    def validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_loss = torch.tensor(val_loss / num_items)

        self.logger.log_hyperparams(
            params={
                "model": self.model_name,
                **self.model_dict,
                "temperature": self.temperature,
                "data": self.trainer.datamodule.json_path,
                "ds_ratio": self.trainer.datamodule.downsample_ratio,
                "batch_size": self.trainer.datamodule.batch_size,
                "distribution": self.trainer.datamodule.dist,
                # "benchmark": self.trainer.benchmark,
                "max_epochs": self.trainer.max_epochs,
                "precision": self.trainer.precision,
            },
            metrics={"contrastive_loss": mean_val_loss},
        )


if __name__ == "__main__":
    cli = LightningCLI(save_config_overwrite=True)
