from fimmia.fimmia_models.registry import register_model
from torch import nn
from dataclasses import dataclass
import torch


@dataclass
class FiMMIAModelArguments:
    model_name: str
    embedding_size: int = 4096
    projection_size: int = 512
    model_path: str = None
    image_embedding_size: int = 1024


@register_model
class BaseLineModelV2(nn.Module):

    def __init__(self, model_args: FiMMIAModelArguments):
        super(BaseLineModelV2, self).__init__()
        embedding_size = model_args.embedding_size
        projection_size = model_args.projection_size
        self.image_embedding_size = model_args.image_embedding_size
        self.projection_size = projection_size
        self.embedding_size = embedding_size
        self.loss_component = nn.Sequential(
            nn.Linear(1, projection_size),
            nn.Dropout(0.2), nn.ReLU()
        )
        self.embedding_component = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(embedding_size // 2, 512),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        self.attack_encoding = nn.Sequential(
            nn.Linear(projection_size * 2, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.ReLU()
        )

    def forward(self, loss_input, embedding_input, labels=None, **kwargs):
        loss_input = torch.as_tensor(loss_input, dtype=torch.float32)
        embedding_input = torch.as_tensor(embedding_input, dtype=torch.float32)
        loss_proj = self.loss_component(loss_input.reshape(-1, 1))
        embed_proj = self.embedding_component(embedding_input)
        proj = torch.hstack([loss_proj, embed_proj])
        res = self.attack_encoding(proj)

        if labels is not None:
            loss = nn.functional.cross_entropy(
                torch.as_tensor(res, dtype=torch.float32),
                torch.as_tensor(labels.view(-1), dtype=torch.int64)
            )
            res = loss, res
        return res


@register_model
class FiMMIAImageAllModelV2(nn.Module):

    def __init__(self, model_args: FiMMIAModelArguments):
        super(FiMMIAImageAllModelV2, self).__init__()
        embedding_size = model_args.embedding_size
        projection_size = model_args.projection_size
        image_embedding_size = model_args.image_embedding_size
        self.projection_size = projection_size
        self.embedding_size = embedding_size
        self.loss_component = nn.Sequential(
            nn.Linear(1, projection_size),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        self.embedding_component = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(embedding_size // 2, 512),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        self.image_embedding_component = nn.Sequential(
            nn.Linear(image_embedding_size, image_embedding_size // 2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(image_embedding_size // 2, 512),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        self.attack_encoding = nn.Sequential(
            nn.Linear(projection_size * 3, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.ReLU(),
        )

    def forward(self, loss_input, embedding_input, image_embedding_input, labels=None, **kwargs):
        loss_input = torch.as_tensor(loss_input, dtype=torch.float32)
        embedding_input = torch.as_tensor(embedding_input, dtype=torch.float32)
        image_embedding_input = torch.as_tensor(image_embedding_input, dtype=torch.float32)

        loss_proj = self.loss_component(loss_input.reshape(-1, 1))
        embed_proj = self.embedding_component(embedding_input)
        image_embed_proj = self.image_embedding_component(image_embedding_input)

        proj = torch.hstack([loss_proj, embed_proj, image_embed_proj])
        res = self.attack_encoding(proj)

        if labels is not None:
            loss = nn.functional.cross_entropy(
                torch.as_tensor(res, dtype=torch.float32),
                torch.as_tensor(labels.view(-1), dtype=torch.int64)
            )
            res = loss, res
        return res


@register_model
class FiMMIAImageAllModelLossNormLinearV2(FiMMIAImageAllModelV2):

    def forward(self, loss_input, embedding_input, image_embedding_input, min_loss=0, loss_diff=1, labels=None):
        loss_input = (loss_input - min_loss) / loss_diff
        return super().forward(loss_input, embedding_input, image_embedding_input, labels)


@register_model
class FiMMIAImageAllModelLossNormSTDV2(FiMMIAImageAllModelV2):

    def forward(self, loss_input, embedding_input, image_embedding_input, mean=0, std=1, labels=None):
        loss_input = (loss_input - mean) / std
        return super().forward(loss_input, embedding_input, image_embedding_input, labels)


@register_model
class FiMMIABaseLineModelLossNormLinearV2(BaseLineModelV2):

    def forward(self, loss_input, embedding_input, min_loss=0, loss_diff=1, labels=None):
        loss_input = (loss_input - min_loss) / loss_diff
        return super().forward(loss_input, embedding_input, labels)


@register_model
class FiMMIABaseLineModelLossNormSTDV2(BaseLineModelV2):

    def forward(self, loss_input, embedding_input, mean=0, std=1, labels=None):
        loss_input = (loss_input - mean) / std
        return super().forward(loss_input, embedding_input, labels)


def cos_sin(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.cos(x), torch.sin(x)], -1)


class Periodic(nn.Module):
    def __init__(self,
                 n_features: int,
                 init_sigma: float = 10.0,
                 initialization: str = "normal",
                 **kwargs) -> None:
        super().__init__()
        if initialization == 'log-linear':
            coefficients = init_sigma ** (torch.arange(n_features) / n_features)
        else:
            assert initialization == 'normal'
            coefficients = torch.normal(0.0, init_sigma, (n_features, ))
        self.coefficients = nn.Parameter(coefficients)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2
        return cos_sin(2 * torch.pi * self.coefficients * x)


@register_model
class FiMMIABaseLineModelLossNormPeriodicV2(BaseLineModelV2):
    def __init__(self, model_args: FiMMIAModelArguments):
        super(FiMMIABaseLineModelLossNormPeriodicV2, self).__init__(model_args)
        self.loss_component = nn.Sequential(
            Periodic(
                self.projection_size // 2,
            ),
            nn.Linear(self.projection_size, self.projection_size),
            nn.Dropout(0.2),
            nn.ReLU()
        )


@register_model
class FiMMIAImageAllModelLossNormPeriodicV2(FiMMIAImageAllModelV2):
    def __init__(self, model_args: FiMMIAModelArguments):
        super(FiMMIAImageAllModelLossNormPeriodicV2, self).__init__(model_args)
        self.loss_component = nn.Sequential(
            Periodic(
                self.projection_size // 2,
            ),
            nn.Linear(self.projection_size, self.projection_size),
            nn.Dropout(0.2),
            nn.ReLU()
        )
