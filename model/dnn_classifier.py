import torch.nn as nn


class DNNClassifier(nn.Module):
    """Feed-forward classifier over shapelet features."""

    def __init__(self, args):
        super().__init__()
        input_dim = args.input_dim
        hidden_dims = args.hidden_dims
        num_classes = args.num_classes

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        """Apply Xavier initialization to linear layers."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)

    def forward(self, x):
        """Run a forward pass."""
        return self.output(self.network(x))
