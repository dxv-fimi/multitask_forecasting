import argparse

import pytorch_lightning as pl
from omegaconf import OmegaConf

from data.precipitation import PrecipationDataset, create_data_loader
from models import *

parser = argparse.ArgumentParser(description="trainer")
parser.add_argument('--config', metavar='cf', type=str, default='config/default.yaml', required=False, help='path to yaml file contain training settings')

model_dict = {
    'ieid': IEID
}

def main():
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)
    module = model_dict[conf.model.module]
    model = module(conf)
    train_dataset = PrecipationDataset(
        data_path=conf.dataset.train,
        win_size=conf.model.data_config.win_size,
        normalize_factor=conf.model.data_config.normalize_factor
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=conf.training.batch_size,
        num_workers=conf.resource.num_workers,
        shuffle=True
    )
    trainer = pl.Trainer(
        gpus=conf.resource.gpus,
        max_epochs=conf.training.max_epochs
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader
    )

if __name__ == '__main__':
    main()
