"""
DL baseline model architectures.
Author: JiaWei Jiang
"""
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    """Naive MLP architecture.

    Parameters:
        t_window: lookback time window
        n_feats: number of features at each time point
        h_dim: hidden dimension of linear layer
        dropout: dropout ratio
    """

    def __init__(self, t_window: int, n_feats: int, h_dim: int, dropout: float):
        self.name = self.__class__.__name__
        super(MLP, self).__init__()

        # Network parameters
        self.t_window = t_window
        self.n_feats = n_feats
        in_dim = t_window * n_feats
        self.h_dim = h_dim
        self.dropout = dropout

        # Model blocks
        self.fc = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.BatchNorm1d(h_dim),  # Seems to pull away prf of different folds
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        x = x.contiguous().view(batch_size, -1)

        output = self.fc(x)
        output = torch.squeeze(output, dim=-1)

        return output


class NaiveRNN(nn.Module):
    """Naive RNN architecture aiming at capturing temporal patterns.

    Parameters:
        t_window: lookback time window
        n_feats: number of features at each time point
        conv_out_ch: number of output channels of temporal convolution
        kernel_size: kernel size of temporal convolution
        rnn_h_dim: hidden dimension of recurrent block
        rnn_n_layers: number of recurrent layers
        rnn_dropout: dropout ratio of recurrent block
        common_dropout: common dropout ratio
    """

    def __init__(
        self,
        t_window: int,
        n_feats: int,
        conv_out_ch: int,
        kernel_size: int,
        rnn_h_dim: int,
        rnn_n_layers: int = 1,
        rnn_dropout: float = 0,
        common_dropout: float = 0.2,
        use_layer_norm: bool = True,
        use_cap: bool = True,
        tconv_act: Optional[str] = None,
    ):
        self.name = self.__class__.__name__
        super(NaiveRNN, self).__init__()

        # Network parameters
        # Temporal convolution
        self.t_window = t_window
        self.n_feats = n_feats
        self.conv_out_ch = conv_out_ch
        self.kernel_size = kernel_size
        self.use_layer_norm = use_layer_norm
        # RNN
        self.rnn_h_dim = rnn_h_dim
        self.rnn_n_layers = rnn_n_layers
        self.rnn_dropout = rnn_dropout
        # Output
        self.use_cap = use_cap
        # Common
        self.common_dropout = common_dropout

        # Model blocks
        # Temporal convolution
        self.tconv = nn.Conv1d(n_feats, conv_out_ch, kernel_size)
        seq_len = t_window - kernel_size + 1
        if tconv_act == "relu":
            self.tconv_act = nn.ReLU()
        elif tconv_act == "tanh":
            self.tconv_act = nn.Tanh()
        else:
            self.tconv_act = None
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(conv_out_ch)
        # RNN
        self.rnn = nn.LSTM(
            conv_out_ch,
            rnn_h_dim,
            num_layers=rnn_n_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        # Output
        ol_dim = rnn_h_dim + 1 if use_cap else rnn_h_dim
        self.output = nn.Linear(ol_dim, 1)

    def forward(self, x: Tensor, cap: Optional[Tensor] = None) -> Tensor:
        """Forward pass.

        Shape:
            x: (B, T, F), where B is the batch size, T is the lookback
                time window, and F is the number of features
            cap: (B, )
        """
        # Temporal convolution
        x = x.transpose(1, 2)
        x = self.tconv(x)  # (B, conv_out_ch, seq_len)
        if self.tconv_act is not None:
            x = self.tconv_act(x)
        x = x.transpose(1, 2)
        if self.use_layer_norm:
            x = self.layer_norm(x)
        # Capacity as a multiplication factor
        #         if cap is not None:
        #             cap = cap[..., None, None]
        #             x = x * cap

        # RNN
        _, (h_n, c_n) = self.rnn(x)
        if self.rnn_n_layers == 1:
            h_n = torch.squeeze(h_n, dim=0)  # (B, rnn_h_dim)
        else:
            h_n = h_n[-1]

        # Output
        if self.use_cap:
            assert cap is not None
            cap = torch.unsqueeze(cap, dim=-1)
            h_n = torch.cat((h_n, cap), dim=1)
        output = self.output(h_n)
        output = torch.squeeze(output, dim=-1)

        return output
