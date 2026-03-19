import torch.nn as nn


class EncoderWithClassifier(nn.Module):
    """Stage 1 model: encoder backbone plus MLP classifier head."""

    def __init__(self, args, encoder_cls):
        super().__init__()

        self.encoder = encoder_cls(
            in_channels=args.in_channels,
            channels=args.channels,
            depth=args.depth,
            reduced_size=args.reduced_size,
            out_channels=args.out_channels,
            kernel_size=args.kernel_size,
        ).to(args.device)

        self.classifier = nn.Sequential(
            nn.Linear(args.out_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, args.num_classes),
        ).to(args.device)

    def forward(self, x):
        encoded = self.encoder(x)
        logits = self.classifier(encoded)
        return logits, encoded
