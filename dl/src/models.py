from torch import nn


class LinearNormAct(nn.Sequential):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.add_module("linear", nn.Linear(in_features, out_features, bias=False))
        self.add_module("norm", nn.LayerNorm(out_features))
        self.add_module("act", nn.ReLU())


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        groups: int = 1,
    ):
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv1d(
                in_features,
                out_features,
                kernel_size,
                groups,
                padding=kernel_size // 2,
                bias=False,
            ),
        )
        self.add_module("norm", nn.BatchNorm1d(out_features))
        self.add_module("act", nn.ReLU())


class ConvNormActPool(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int = 3,
        groups: int = 1,
    ):
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv1d(
                in_features,
                out_features,
                kernel_size,
                groups,
                padding=kernel_size // 2,
                bias=False,
            ),
        )
        self.add_module("norm", nn.BatchNorm1d(out_features))
        self.add_module("act", nn.ReLU())
        self.add_module("pool", nn.MaxPool1d(2, 2, padding=0))


class CodeClassificationMLP(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int,
        num_classes: int,
        hidden_dims: list[int] = [256],
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.layers = nn.Sequential()
        prev_dim = embedding_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.add_module(f"block_{i}", LinearNormAct(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.layers(self.embedding(x))
        x = self.pool(x.movedim(1, -1)).squeeze(-1)
        return self.classifier(x)


class CodeClassificationCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int,
        num_classes: int,
        hidden_dims: list[int] = [256],
        kernel_sizes: list[int] = [5],
        block: nn.Module = ConvNormAct,
    ) -> None:
        super().__init__()
        assert len(kernel_sizes) == len(
            hidden_dims
        ), f"len(kernel_sizes) and len(hidden_dims) must be same, got {len(kernel_sizes)=}, {len(hidden_dims)=}"
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.layers = nn.Sequential()
        prev_dim = embedding_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.add_module(
                f"block_{i}", block(prev_dim, hidden_dim, kernel_sizes[i])
            )
            prev_dim = hidden_dim
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x).movedim(1, -1)
        x = self.layers(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)
