from typing import List

import torch
from torch import nn


class IrisClassifier(nn.Module):  # type: ignore
    def __init__(self) -> None:
        super(IrisClassifier, self).__init__()
        # Parameters
        self.learning_rate = 0.01

        # Define the layers
        self.h1_layer = nn.Linear(4, 3)
        self.softmax = nn.Softmax(dim=0)

        # Define loss functions
        self.loss = nn.BCELoss()

        # Define optimizer
        self.optimizer = torch.optim.SGD(
            params=self.parameters(), lr=self.learning_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.h1_layer(x)
        y_hat = self.softmax(h)
        return y_hat

    def backward(self, y_hat: torch.Tensor, y: torch.Tensor) -> float:
        self.optimizer.zero_grad()

        # Calculate the loss
        loss = self.loss(y_hat, y)

        # Backward
        loss.backward()

        # Update parameter
        self.optimizer.step()
        return float(loss.data.item())

    def predict(self, x: torch.Tensor) -> List[float]:
        y_hat = self.forward(x)
        prediction: List[float] = y_hat.tolist()
        return prediction
