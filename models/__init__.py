from typing import Dict, List, Union

import torch
from torch.utils.data.dataloader import DataLoader

nn = torch.nn
import math
from copy import deepcopy

import pytorch_lightning as pl


def create_lstm_encoder(config: Dict):
    config['input_size'] = config.get('input_size', 1)
    config['hidden_size'] = config.get('hidden_size', config['input_size'])
    config['num_layers'] = config.get('num_layers', 1)
    return LSTMEncoder(**config), config.get('proj_size', 0) if config.get('proj_size', 0) > 0 else config['hidden_size']

def create_mlp_encoder(config: Dict):
    config['input_size'] = config.get('input_size', 1)
    config['hidden_size'] = config.get('hidden_size', config['input_size'])
    config['output_size'] = config.get('output_size', config['hidden_size'][-1]) \
        if type(config['hidden_size'] == list) else config['hidden_size']
    config['num_hidden_layer'] = config.get('num_hidden_layer', 1)
    return MLPEncoder(**config), config['output_size']

def create_mlp_joint_net(config: Dict):
    config['num_stations'] = config.get('num_stations', 1)
    config['input_size'] = config.get('input_size', [1] * config['num_stations'])
    config['hidden_size'] = config.get('hidden_size', sum(config['input_size']))
    _last_hidden_size = config['hidden_size'][-1] \
        if type(config['hidden_size']) == list \
        else config['hidden_size']
    config['output_size'] = config.get('output_size', \
        [math.ceil(_last_hidden_size / config['num_stations'])] * config['num_stations'])
    config['num_hidden_layer'] = config.get('num_hidden_layer', 1)
    return MLPJointNet(**config), config['output_size']

def create_mlp_decoder(config: Dict):
    config['input_size'] = config.get('input_size', 1)
    config['num_hidden_layer'] = config.get('num_hidden_layer', 1)
    config['hidden_size'] = config.get('hidden_size', config['input_size'])
    return MLPDecoder(**config)

def sum_max_loss(base_loss, alpha, *args, **kwargs):
    base_loss = loss_functions[base_loss.func](**base_loss.params)
    def cal(y_hat, y):
        loss = base_loss(y_hat, y)
        loss = alpha * torch.sum(loss) + (1 - alpha) * loss.size(0) * torch.max(loss)
        return loss
    return cal

creators = {
    'lstm_encoder': create_lstm_encoder,
    'mlp_encoder': create_mlp_encoder,
    'mlp_joint_net': create_mlp_joint_net,
    'mlp_decoder': create_mlp_decoder
}

loss_functions = {
    'sum_max_loss': sum_max_loss,
    'MSELoss': torch.nn.MSELoss
}

def create_module(config):
    creator = creators[config['module']]
    return creator(config['params'])

def create_loss(config):
    creator = loss_functions[config.func]
    return creator(**config.params)

class IEID(pl.LightningModule):

    def __init__(self, config) -> None:

        self.save_hyperparameters()

        super().__init__()

        self.config = config
        self.num_stations = num_stations = config.model.data_config.num_stations

        encoder_config = deepcopy(config.model.encoder_config)
        decoder_config = deepcopy(config.model.decoder_config)
        joint_net_config = deepcopy(config.model.joint_net_config)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.encoders_size = []
        self.encoding_size = []

        for _ in range(num_stations):
            encoder, encoding_size = create_module(encoder_config)
            self.encoders.append(encoder)
            self.encoding_size.append(encoding_size)

        joint_net_config.params.num_stations = self.num_stations
        joint_net_config.params.input_size = self.encoding_size
        self.joint_net, dec_inp_size = create_module(joint_net_config)

        for i in range(num_stations):
            decoder_config.params.input_size = dec_inp_size[i]
            self.decoders.append(create_module(decoder_config))

        self.loss = create_loss(config.training.loss)

    def forward(self, input):
        """
            input -- tensor of shape (batch_size, win_size, num_stations, 1)
        """
        enc_out = []
        for i in range(self.num_stations):
            encoder = self.encoders[i]
            dec_inp = input[:, :, i]
            enc_out.append(encoder.encode(dec_inp))
        dec_inp = self.joint_net.join(enc_out)
        output = []
        for i in range(self.num_stations):
            decoder = self.decoders[i]
            output.append(decoder.decode(dec_inp[i]))
        return torch.cat(output, -1)

    def training_step(self, batch, batch_idx):
        x = batch['x']
        y = batch['y']
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("avg_loss", loss / self.num_stations, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def config_training(self, loss):
        self.set_loss(loss)
    
    def set_loss(self, loss):
        self.loss = loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    

class LSTMEncoder(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.encoder = nn.LSTM(*args, **kwargs)
        self.hidden_state = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.hidden_state == None:
            self.init_hidden_state(input.size(0))
        self.hidden_state = (
            self.hidden_state[0].type_as(input),
            self.hidden_state[1].type_as(input)
        )
        output, self.hidden_state = self.encoder(input, self.hidden_state)
        return output, self.hidden_state

    def encode(self, input: torch.Tensor, reset_hidden: bool = True) -> torch.Tensor:
        if reset_hidden:
            self.init_hidden_state(input.size(0))
        output, _ = self(input)
        # Lay output cua step cuoi cung
        if self.encoder.batch_first:
            output = output[:, -1]
        else:
            output = output[-1, :]
        return output

    def init_hidden_state(self, batch_size):
        encoder = self.encoder
        hidden_state = torch.zeros(encoder.num_layers, batch_size, encoder.hidden_size)
        cell_state = torch.zeros(encoder.num_layers, batch_size, encoder.hidden_size)
        self.hidden_state = (hidden_state, cell_state)


class MLPEncoder(pl.LightningModule):
    def __init__(self,
    input_size: int,
    num_hidden_layer: int,
    hidden_size: Union[int, List[int]],
    output_size: int,
    *args, **kwargs) -> None:
        super().__init__()
        seq = []
        if type(hidden_size) != list:
            hidden_size = [hidden_size] * num_hidden_layer
            hidden_size.append(output_size)
        else:
            hidden_size = hidden_size + [output_size]
        prev_size = input_size
        for i in range(num_hidden_layer + 1):
            seq.append(nn.Linear(prev_size, hidden_size[i]))
            prev_size = hidden_size[i]
            if i < num_hidden_layer:
                seq.append(nn.LeakyReLU())
        self.encoder = nn.Sequential(*seq)
        self.input_size = input_size
        self.num_hidden_layer = num_hidden_layer
        self.hidden_size = hidden_size[:-1]
        self.output_size = output_size

    def forward(self, input):
        return self.encoder(input)

    def encoder(self, input, *args, **kwargs):
        return self(input)

class MLPJointNet(pl.LightningModule):
    def __init__(self,
    input_size: List[int],
    num_hidden_layer: int,
    hidden_size: Union[int, List[int]],
    output_size: List[int],
    *args, **kwargs):
        super().__init__()
        seq = []
        if type(hidden_size) != list:
            hidden_size = [hidden_size] * num_hidden_layer
            hidden_size.append(sum(output_size))
        else:
            hidden_size = hidden_size + [sum(output_size)]
        prev_size = sum(input_size)
        for i in range(num_hidden_layer + 1):
            seq.append(nn.Linear(prev_size, hidden_size[i]))
            prev_size = hidden_size[i]
            if i < num_hidden_layer:
                seq.append(nn.LeakyReLU())
        self.joint_net = nn.Sequential(*seq)
        self.input_size = input_size
        self.num_hidden_layer = num_hidden_layer
        self.hidden_size = hidden_size[:-1]
        self.output_size = output_size

    def forward(self, input):
        return self.joint_net(input)

    def join(self, input):
        jn_out = self(torch.cat(input, -1))
        output = []
        idx = 0
        for i in self.output_size:
            output.append(jn_out[:, idx:idx+i])
            idx += i
        return output

class MLPDecoder(pl.LightningModule):
    def __init__(self, input_size: int, num_hidden_layer: int, hidden_size: Union[int, List[int]], *args, **kwargs):
        super().__init__()
        seq = []
        if type(hidden_size) != list:
            hidden_size = [hidden_size] * num_hidden_layer
            hidden_size.append(1)
        else:
            hidden_size = hidden_size + [1]
        prev_size = input_size
        for i in range(num_hidden_layer + 1):
            seq.append(nn.Linear(prev_size, hidden_size[i]))
            prev_size = hidden_size[i]
            if i < num_hidden_layer:
                seq.append(nn.LeakyReLU())
        self.decoder = nn.Sequential(*seq)
        self.input_size = input_size
        self.num_hidden_layer = num_hidden_layer
        self.hidden_size = hidden_size[:-1]

    def forward(self, input):
        return self.decoder(input)

    def decode(self, input):
        return self(input)
