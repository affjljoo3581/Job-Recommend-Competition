from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class SiDConfig:
    num_categories_list: list[int]
    num_numerical_features: int
    num_text_features: int
    text_input_size: int
    embedding_size: int = 64
    hidden_size: int = 256
    intermediate_size: int = 1024
    num_hidden_layers: int = 18
    num_transform_blocks: int = 1
    num_attention_blocks: int = 1
    hidden_dropout_prob: float = 0.5
    attention_dropout_prob: float = 0.5
    drop_path_prob: float = 0.5
    embed_init_std: float = 0.02
    num_labels: Optional[int] = None

    @property
    def num_total_categories(self) -> int:
        return sum(self.num_categories_list)

    @property
    def num_categorical_features(self) -> int:
        return len(self.num_categories_list)

    @property
    def num_total_features(self) -> int:
        return (
            self.num_categorical_features
            + self.num_numerical_features
            + self.num_text_features
        )

    @property
    def total_embedding_size(self) -> int:
        return self.num_total_features * self.embedding_size


class StochasticDepth(nn.Module):
    def __init__(self, drop_path_prob: float = 0.1):
        super().__init__()
        self.drop_path_prob = drop_path_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.drop_path_prob == 0 or not self.training:
            return hidden_states

        mask = torch.rand((hidden_states.size(0), 1), device=hidden_states.device)
        mask = (mask > self.drop_path_prob).type_as(hidden_states) / self.drop_path_prob
        return mask * hidden_states


class SiDEmbeddings(nn.Module):
    def __init__(self, config: SiDConfig):
        super().__init__()
        self.categorical_embeddings = nn.Embedding(
            config.num_total_categories, config.embedding_size
        )
        self.numerical_direction = nn.Parameter(
            torch.rand(config.num_numerical_features, config.embedding_size)
        )
        self.numerical_anchor = nn.Parameter(
            torch.rand(config.num_numerical_features, config.embedding_size)
        )

        # Although we define the multiple dense layers to project each text embedding to
        # the input embedding space, we will use batched (stacked) matmul by gathering
        # the weight matrices.
        self.text_projections = nn.ModuleList(
            nn.Linear(config.text_input_size, config.embedding_size, bias=False)
            for _ in range(config.num_text_features)
        )

        # Create embedding offsets which indicate the start embedding index of each
        # categorical feature. Because this class uses only one embedding layer which
        # contains the embedding vectors for all categorical features, it is necessary
        # to separate each embedding group from other features.
        self.register_buffer(
            "categorical_embedding_offsets",
            torch.tensor([[0] + config.num_categories_list[:-1]]).cumsum(1),
        )

    def forward(
        self,
        categorical_inputs: torch.Tensor,
        numerical_inputs: torch.Tensor,
        text_inputs: torch.Tensor,
    ) -> torch.Tensor:
        # Add embedding offsets to the categorical features to map to the corresponding
        # embedding groups.
        categorical_inputs = categorical_inputs + self.categorical_embedding_offsets
        categorical_embeddings = self.categorical_embeddings(categorical_inputs)

        numerical_embeddings = numerical_inputs[:, :, None] * self.numerical_direction
        numerical_embeddings = numerical_embeddings + self.numerical_anchor

        stacked_weight = torch.stack(
            [layer.weight.transpose(0, 1) for layer in self.text_projections],
        )
        text_embeddings = torch.einsum("btm,tmn->btn", text_inputs, stacked_weight)

        # After creating embedding vectors for categorical, numerical and text features,
        # they will be concatenated to a single tensor.
        return torch.cat(
            (categorical_embeddings, numerical_embeddings, text_embeddings), dim=1
        )


class SiDResidualBlock(nn.Module):
    def __init__(self, config: SiDConfig):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size * 2),
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.feedforward(hidden_states)

        output, gating = output.chunk(2, dim=1)
        output = output * gating.sigmoid()

        return hidden_states + self.dropout(output)


class SiDLayer(nn.Module):
    def __init__(self, config: SiDConfig, use_attention: bool = True):
        super().__init__()
        if use_attention:
            self.attention = nn.Sequential(
                *[SiDResidualBlock(config) for _ in range(config.num_attention_blocks)],
                nn.Linear(config.hidden_size, config.num_total_features),
                nn.Sigmoid(),
            )
            self.dropout = nn.Dropout(config.attention_dropout_prob)

        self.projection = nn.Linear(config.total_embedding_size, config.hidden_size)
        self.transform = nn.Sequential(
            *[SiDResidualBlock(config) for _ in range(config.num_transform_blocks)]
        )
        self.droppath = StochasticDepth(config.drop_path_prob)

    def forward(
        self,
        input_embeddings: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Calculate the attention probabilities and multiply to the embeddings.
        if hasattr(self, "attention") and hidden_states is not None:
            attention_probs = self.attention(hidden_states)
            attention_probs = self.dropout(attention_probs)
            input_embeddings = input_embeddings * attention_probs[:, :, None]

        output = self.projection(input_embeddings.flatten(1))
        output = self.transform(output)

        # If `hidden_states` is not None then use residual connection.
        if hidden_states is not None:
            return hidden_states + self.droppath(output)
        return output


class SiDModel(nn.Module):
    def __init__(self, config: SiDConfig):
        super().__init__()
        self.config = config
        self.embeddings = SiDEmbeddings(config)
        self.layers = nn.ModuleList(
            SiDLayer(config, use_attention=i > 0)
            for i in range(config.num_hidden_layers)
        )
        self.normalization = nn.LayerNorm(config.hidden_size)
        self.init_weights()

    def init_weights(self, module: Optional[nn.Module] = None):
        if module is None:
            self.apply(self.init_weights)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.embed_init_std)
        elif isinstance(module, SiDEmbeddings):
            nn.init.normal_(module.numerical_direction, std=self.config.embed_init_std)
            nn.init.normal_(module.numerical_anchor, std=self.config.embed_init_std)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, 5 ** 0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        categorical_inputs: torch.Tensor,
        numerical_inputs: torch.Tensor,
        text_inputs: torch.Tensor,
    ) -> torch.Tensor:
        input_embeddings = self.embeddings(
            categorical_inputs,
            numerical_inputs,
            text_inputs,
        )

        hidden_states = None
        for layer in self.layers:
            hidden_states = layer(input_embeddings, hidden_states)

        hidden_states = self.normalization(hidden_states)
        return hidden_states


class SiDClassifier(nn.Module):
    def __init__(self, config: SiDConfig):
        super().__init__()
        self.model = SiDModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        categorical_inputs: torch.Tensor,
        numerical_inputs: torch.Tensor,
        text_inputs: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(categorical_inputs, numerical_inputs, text_inputs)
        logits = self.classifier(hidden_states)
        return logits
