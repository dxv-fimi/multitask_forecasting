import argparse

from omegaconf import OmegaConf

from data.precipitation import PrecipationDataset
from models import *

parser = argparse.ArgumentParser(description="trainer")
parser.add_argument('--checkpoint', metavar='-cp', type=str, default='checkpoints/default/', required=False, help='path to folder contain hparams.yaml and checkpoint.ckpt file')

model_dict = {
    'ieid': IEID
}

def main():
    args = parser.parse_args()
    conf = OmegaConf.load(args.checkpoint + 'hparams.yaml').config
    module = model_dict[conf.model.module]
    model = module.load_from_checkpoint(args.checkpoint + 'checkpoint.ckpt')
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
