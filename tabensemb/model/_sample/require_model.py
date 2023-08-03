from typing import List, Dict
from ..base import get_linear, AbstractNN
import torch
from torch import nn
import numpy as np
from tabensemb.model.base import get_sequential, AbstractWrapper


class RequireOthersNN(AbstractNN):
    def __init__(self, datamodule, required_models, **kwargs):
        super(RequireOthersNN, self).__init__(datamodule, **kwargs)
        self.required_model = list(required_models.values())[0]
        self.required_model_name = list(required_models.keys())[0]

        self.use_hidden_rep, hidden_rep_dim = self._test_required_model(
            self.n_inputs, self.required_model
        )
        if self.use_hidden_rep:
            self.head = get_sequential(
                [128, 64, 32],
                n_inputs=hidden_rep_dim,
                n_outputs=1,
                act_func=nn.ReLU,
                dropout=0,
                use_norm=False,
            )
        else:
            # If no trainable param in nn.Module, loss.backward() crashes.
            self.dummy_var = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

    def _forward(
        self, x: torch.Tensor, derived_tensors: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pred = self.call_required_model(
            self.required_model,
            x,
            derived_tensors,
            model_name=self.required_model_name,
        )
        if self.use_hidden_rep:
            hidden = self.get_hidden_state(
                self.required_model,
                x,
                derived_tensors,
                model_name=self.required_model_name,
            )
            return self.head(hidden)
        else:
            if self.training:
                return pred * self.dummy_var
            else:
                return pred
