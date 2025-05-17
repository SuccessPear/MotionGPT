# train_projection_vi_pl.py

import os
import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import MT5Tokenizer, MT5EncoderModel, MT5ForConditionalGeneration
from mGPT.config import parse_args
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.load_checkpoint import load_pretrained_vae

def main():
    # === Configuration ===
    cfg = parse_args(phase="train")
    # cfg.DATASET.HUMANML3D.TEXT_DIR = "./datasets/humanml3d/texts_vi"
    # cfg.DATASET.HUMANML3D.MOTION_DIR = "./datasets/humanml3d/new_joint_vecs"
    # cfg.DATASET.HUMANML3D.SPLIT_FILE = "./datasets/humanml3d/train.txt"

    # === DataModule ===
    datamodule = build_data(cfg)
    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    print(f"Number of batches {len(train_loader)}")

    # === Build Pretrained Model ===
    model = build_model(cfg, datamodule)
    load_pretrained_vae(cfg, model, logger=None)  # Only VAE is needed
    motion_decoder = model.vae.decoder

    # === Dataset Wrapper ===
    class HumanML3DDataset(Dataset):
        def __init__(self, dataloader):
            self.data = list(dataloader.dataset)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]['text'], self.data[idx]['motion']

    def collate_fn(batch):
        texts, motions = zip(*batch)
        padded_motions = torch.nn.utils.rnn.pad_sequence(motions, batch_first=True)
        return list(texts), padded_motions

    # === Lightning Module ===
    class ProjectionTrainer(pl.LightningModule):
        def __init__(self, decoder):
            super().__init__()
            self.tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
            self.encoder = MT5EncoderModel.from_pretrained("google/mt5-base")
            for p in self.encoder.parameters():
                p.requires_grad = False

            self.projection = torch.nn.Sequential(
                torch.nn.Linear(768, 512),
                torch.nn.GELU(),
                torch.nn.Linear(512, 512),
                torch.nn.LayerNorm(512)
            )

            self.decoder = decoder
            for p in self.decoder.parameters():
                p.requires_grad = False

            self.loss_fn = torch.nn.MSELoss()
            self.to_motion = torch.nn.Linear(512, 263)

        def forward(self, texts):
            tokens = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)
            with torch.no_grad():
                emb = self.encoder(**tokens).last_hidden_state[:, 0, :]  # [B, 768]
            latent = self.projection(emb)  # [B, 512]
            #print("haha: ", latent.shape)
            return self.decoder(latent.unsqueeze(-1))
            #return self.decoder(latent.unsqueeze(-1))  # [B, T, 263]

        def training_step(self, batch, batch_idx):
            #texts, motions = batch
            motions = batch['motion']
            texts = batch['text']
            motions = motions.to(self.device)
            preds = self(texts)
            #print(preds.shape)
            #print("motion: ", motions.shape)
            loss = self.loss_fn(preds, motions.permute(0,2,1)[:, :, :preds.shape[2]])
            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.projection.parameters(), lr=2e-5)

    # === Prepare Dataset and Trainer ===
    # train_dataset = HumanML3DDataset(datamodule.train_dataloader())
    # dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    model = ProjectionTrainer(motion_decoder)

    trainer = pl.Trainer(
        default_root_dir="./logs/projection_train",
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
    )

    # === Train ===
    trainer.fit(model, datamodule=datamodule)

    # === Save Projection Layer ===
    torch.save(model.projection.state_dict(), "projection_layer.pth")
    print("✅ Projection layer saved as 'projection_layer.pth'")


if __name__ == "__main__":
    main()