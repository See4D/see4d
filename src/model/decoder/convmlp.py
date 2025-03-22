import torch
import torch.nn as nn

class VanillaConvMLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dim_hidden: int, num_layers: int =1, use_bias: bool = False):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = dim_hidden, num_layers
        layers = [
            self.make_linear(
                dim_in,
                self.n_neurons,
                is_first=True,
                is_last=False,
                bias=use_bias,
            ),
            self.make_activation(),
        ]
        for i in range(self.n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    self.n_neurons,
                    self.n_neurons,
                    is_first=False,
                    is_last=False,
                    bias=use_bias,
                ),
                self.make_activation(),
            ]
        layers += [
            self.make_linear(
                self.n_neurons,
                dim_out,
                is_first=False,
                is_last=True,
                bias=use_bias,
            )
        ]
        self.layers = nn.Sequential(*layers)
        self.output_activation = nn.Identity()

    def forward(self, x):
        # disable autocast
        # strange that the parameters will have empty gradients if autocast is enabled in AMP
        with torch.cuda.amp.autocast(enabled=False):
            x = self.layers(x)
            x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last, bias=False):
        # layer = nn.Linear(dim_in, dim_out, bias=bias)
        layer = nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)
        return layer

    def make_activation(self):
        return nn.ReLU(inplace=True)
