from typing import List, Dict
from ..base import get_linear, AbstractNN
import torch
from torch import nn
import numpy as np
from tabensemb.model.base import get_sequential


class Embedding2d(nn.Module):
    def __init__(
        self,
        embedding_dim,
        embed_dropout,
        cat_num_unique,
        n_inputs,
        run_cat,
    ):
        super(Embedding2d, self).__init__()
        # Module: Continuous embedding
        d_sqrt_inv = 1 / np.sqrt(embedding_dim)
        self.embedding_dim = embedding_dim
        self.cont_norm = nn.BatchNorm1d(n_inputs)
        self.cont_embed_weight = nn.Parameter(torch.Tensor(n_inputs, embedding_dim))
        nn.init.normal_(
            self.cont_embed_weight,
            std=d_sqrt_inv,
        )
        self.cont_embed_bias = nn.Parameter(torch.Tensor(n_inputs, embedding_dim))
        nn.init.normal_(
            self.cont_embed_bias,
            std=d_sqrt_inv,
        )
        self.cont_dropout = nn.Dropout(embed_dropout)

        # Module: Categorical embedding
        if run_cat:
            # See pytorch_widedeep.models.tabular.embeddings_layers.SameSizeCatEmbeddings
            self.cat_embeds = nn.ModuleList(
                [
                    nn.Embedding(
                        num_embeddings=num_unique + 1,
                        embedding_dim=embedding_dim,
                    )
                    for num_unique in cat_num_unique
                ]
            )
            for embed in self.cat_embeds:
                nn.init.normal_(
                    embed.weight,
                    std=d_sqrt_inv,
                )
            self.cat_dropout = nn.Dropout(embed_dropout)
            self.run_cat = True
        else:
            self.run_cat = False

    def forward(self, x, derived_tensors):
        x_cont = self.cont_embed_weight.unsqueeze(0) * self.cont_norm(x).unsqueeze(
            2
        ) + self.cont_embed_bias.unsqueeze(0)
        x_cont = self.cont_dropout(x_cont)
        if self.run_cat:
            cat = derived_tensors["categorical"].long()
            x_cat_embeds = [
                self.cat_embeds[i](cat[:, i]).unsqueeze(1) for i in range(cat.size(1))
            ]
            x_cat = torch.cat(x_cat_embeds, 1)
            x_cat = self.cat_dropout(x_cat)
            x_res = torch.cat([x_cont, x_cat], dim=1)
        else:
            x_res = x_cont
        return x_res


class Embedding1d(nn.Module):
    def __init__(
        self,
        embedding_dim,
        embed_dropout,
        cat_num_unique,
        n_inputs,
        run_cat,
    ):
        super(Embedding1d, self).__init__()
        d_sqrt_inv = 1 / np.sqrt(embedding_dim)
        if run_cat:
            # See pytorch_widedeep.models.tabular.embeddings_layers.SameSizeCatEmbeddings
            self.cat_embeds = nn.ModuleList(
                [
                    nn.Embedding(
                        num_embeddings=num_unique + 1,
                        embedding_dim=embedding_dim,
                    )
                    for num_unique in cat_num_unique
                ]
            )
            # No special initialization in pytorch_tabular.
            # for embed in self.cat_embeds:
            #     nn.init.normal_(
            #         embed.weight,
            #         std=d_sqrt_inv,
            #     )
            self.run_cat = True
        else:
            self.run_cat = False
        self.dropout = nn.Dropout(embed_dropout)
        self.bn = nn.BatchNorm1d(n_inputs)

    def forward(self, x, derived_tensors):
        x_res = self.bn(x)
        if self.run_cat:
            cat = derived_tensors["categorical"].long()
            x_cat_embeds = [self.cat_embeds[i](cat[:, i]) for i in range(cat.size(1))]
            x_cat = torch.cat(x_cat_embeds, 1)
            x_res = torch.cat([x_res, x_cat], dim=1)
        x_res = self.dropout(x_res)
        return x_res


class CategoryEmbeddingNN(AbstractNN):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        datamodule,
        cat_num_unique: List[int] = None,
        embed_extend_dim: bool = False,
        **kwargs,
    ):
        super(CategoryEmbeddingNN, self).__init__(datamodule, **kwargs)

        run_cat = "categorical" in self.derived_feature_names
        self.embed_extend_dim = embed_extend_dim
        if embed_extend_dim:
            linear_input = (
                n_inputs + len(cat_num_unique) * run_cat
            ) * self.hparams.embedding_dim
        else:
            linear_input = (
                n_inputs + len(cat_num_unique) * self.hparams.embedding_dim * run_cat
            )
        self.linear = get_sequential(
            [128, 64],
            n_inputs=linear_input,
            n_outputs=32,
            act_func=nn.ReLU,
            dropout=self.hparams.mlp_dropout,
            use_norm=False,
            out_activate=True,
            out_norm_dropout=True,
        )

        if embed_extend_dim:
            self.embed = Embedding2d(
                self.hparams.embedding_dim,
                self.hparams.embed_dropout,
                cat_num_unique,
                n_inputs,
                run_cat=run_cat,
            )
        else:
            self.embed = Embedding1d(
                self.hparams.embedding_dim,
                self.hparams.embed_dropout,
                cat_num_unique,
                n_inputs,
                run_cat=run_cat,
            )

        self.head = get_linear(
            n_inputs=32,
            n_outputs=n_outputs,
            nonlinearity="relu",
        )
        self.hidden_rep_dim = 32
        self.hidden_representation = None
        # Just for a unit test
        self.not_require_grad = nn.Parameter(torch.zeros((3, 3)), requires_grad=False)

    def _forward(
        self, x: torch.Tensor, derived_tensors: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        x_embed = self.embed(x, derived_tensors)
        if self.embed_extend_dim:
            x_embed = x_embed.flatten(start_dim=1)
        output = self.linear(x_embed)
        self.hidden_representation = output
        output = self.head(output)
        return output
