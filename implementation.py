import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from typing import List

class ByzantineResilientSGD:
    def __init__(self, model: nn.Module, lr: float, tolerance: float = 0.25):
        self.model = model
        self.lr = lr
        self.tolerance = tolerance  # Fraction of Byzantine workers tolerated

    def robust_mean(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        """
        Robust mean estimation using outlier filtering.
        Based on Steinhardt et al. (ITCS 2018).
        """
        gradients = torch.stack(gradients)
        mean = torch.mean(gradients, dim=0)
        deviation = torch.norm(gradients - mean, dim=1)
        threshold = torch.median(deviation) + 1.5 * torch.std(deviation)
        filtered_gradients = gradients[deviation <= threshold]
        return torch.mean(filtered_gradients, dim=0)

    def step(self, worker_gradients: List[torch.Tensor]):
        """
        Perform a single step of Byzantine-resilient SGD.
        """
        robust_gradient = self.robust_mean(worker_gradients)
        with torch.no_grad():
            for param in self.model.parameters():
                param -= self.lr * robust_gradient

    def train(self, worker_gradients_list: List[List[torch.Tensor]], epochs: int):
        """
        Train the model using Byzantine-resilient SGD.
        """
        for epoch in range(epochs):
            for worker_gradients in worker_gradients_list:
                self.step(worker_gradients)

if __name__ == '__main__':
    # Dummy data for testing
    torch.manual_seed(42)
    np.random.seed(42)

    # Define a simple model
    model = nn.Linear(10, 1)

    # Simulate worker gradients
    num_workers = 10
    num_byzantine = 2
    true_gradient = torch.randn(10)  # True gradient
    worker_gradients = [true_gradient + 0.1 * torch.randn(10) for _ in range(num_workers - num_byzantine)]
    
    # Add Byzantine workers with arbitrary gradients
    for _ in range(num_byzantine):
        worker_gradients.append(torch.randn(10) * 10)

    # Initialize Byzantine-resilient SGD
    br_sgd = ByzantineResilientSGD(model, lr=0.01)

    # Train for a few epochs
    worker_gradients_list = [worker_gradients for _ in range(5)]  # Simulate multiple steps
    br_sgd.train(worker_gradients_list, epochs=5)

    # Print model parameters after training
    for param in model.parameters():
        print(param)