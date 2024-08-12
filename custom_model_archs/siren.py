import torch
import numpy as np
from torch import nn
from typing import Union, Tuple, Any, List

from ray.rllib.utils.typing import TensorType

class SIREN(nn.Module):
    '''
    Pytorch layer 
    '''
    """Simple PyTorch version of `linear` function"""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        initializer: Any = None,
        use_bias: bool = True,
        bias_init: float = 0.0,
    ):
        """Creates a SIREN layer, standard linear layer w/ bias
        and a sine activation function

        Args:
            in_size: Input size for SIREN Layer
            out_size: Output size for SIREN Layer
            initializer: Initializer function for FC layer weights
            use_bias: Whether to add bias weights or not
            bias_init: Initalize bias weights to bias_init const
        """
        super(SIREN, self).__init__()
        layers = []

        # Actual nn.Linear layer (including correct initialization logic).
        linear = nn.Linear(in_size, out_size, bias=True)
        if initializer is None:
            initializer = nn.init.xavier_uniform_
        initializer(linear.weight)
        if use_bias is True:
            nn.init.constant_(linear.bias, bias_init)

        #linear layer
        layers.append(linear)
        # Put everything in sequence.
        self._model = nn.Sequential(*layers)

    def forward(self, x: TensorType) -> TensorType:
        #return sine function applied to linear layer
        return torch.sin(self._model(x))