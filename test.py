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
    model.eval()
    test_dataset = PrecipationDataset(
        data_path=conf.dataset.test,
        win_size=conf.model.data_config.win_size,
        normalize_factor=conf.model.data_config.normalize_factor
    )
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=1)
    Y_hat = []
    Y = []

    with torch.no_grad():
        for i, sample in enumerate(test_dataloader):
            x = sample['x']
            Y_hat.append(model(x))
            Y.append(sample['y'])

    Y = torch.cat(Y, dim=0) / conf.model.data_config.normalize_factor
    Y_hat = torch.cat(Y_hat, dim=0) / conf.model.data_config.normalize_factor

    mse_loss = torch.nn.MSELoss()
    print('MSE:', mse_loss(Y_hat, Y).item())
    mae_loss = torch.nn.L1Loss()
    print('MAE:', mae_loss(Y_hat, Y).item())


if __name__ == '__main__':
    main()
