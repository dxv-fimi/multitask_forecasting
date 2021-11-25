import argparse

from omegaconf import OmegaConf

from data.aqi_2 import create_data_loader
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
    train_dataloader, test_dataloader = create_data_loader(
        data_path=conf.dataset.data,
        win_size=conf.model.data_config.win_size,
        batch_size=conf.training.batch_size,
        num_workers=conf.resource.num_workers,
        normalization_factor=conf.model.data_config.normalize_factor
    )
    Y_hat = []
    Y = []

    with torch.no_grad():
        for i, sample in enumerate(test_dataloader):
            x = sample['x']
            Y_hat.append(model(x))
            Y.append(sample['y'])

    Y = torch.cat(Y, dim=0) / conf.model.data_config.normalize_factor
    Y_hat = torch.cat(Y_hat, dim=0) / conf.model.data_config.normalize_factor

    print(Y[:10])
    print(Y[:10])
    mse_loss = torch.nn.MSELoss()
    print('MSE:', mse_loss(Y_hat, Y).item())
    mae_loss = torch.nn.L1Loss()
    print('MAE:', mae_loss(Y_hat, Y).item())
    print('MAE 0', mae_loss(Y_hat[:, 0], Y[:, 0]).item())
    print('MAE 1', mae_loss(Y_hat[:, 1], Y[:, 1]).item())
    
    print('MAPE', mape_loss(Y_hat, Y).item())
    print('MAPE 0', mape_loss(Y_hat[:, 0], Y[:, 0]).item())
    print('MAPE 1', mape_loss(Y_hat[:, 1], Y[:, 1]).item())
    
def mape_loss(y_hat, y):
    return torch.mean(torch.abs((y_hat - y) / y))


if __name__ == '__main__':
    main()
