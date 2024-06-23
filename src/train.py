import torch
import pandas as pd
import pyro
import numpy as np
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from src.pyro_model.ctr.model_ctr import model_ctr, guide_ctr


p = 0.1
length = 1000
def generate_random():
    return np.random.choice([0.0, 1.0], p=[1-p, p])

## テスト
random_numbers = [generate_random() for _ in range(length)]
print(random_numbers)
print(np.mean(random_numbers))

# データをテンソルに変換
data = torch.tensor(random_numbers, dtype=torch.int64)
data = data.long()
print(data)
print(data.dtype)


def train(
    num_iterations: int = 1000, adam_params={"lr": 0.005, "betas": (0.90, 0.999)}
) -> None:
    """ """
  
    pyro.clear_param_store()
    # Setting optimizer
    optimizer = Adam(adam_params)

    # Setting ELBO
    svi = SVI(model_ctr, guide_ctr, optimizer, loss=Trace_ELBO())

    # モデルのトレーニング
    for i in range(num_iterations):
        loss = svi.step(data.float())
        if i % 100 == 0:
            print(f"Iteration {i} : loss = {loss}")

    # CTRの事後分布を確認

    theoretical_mean, theoretical_variance = evaluate_model_theoretical()
    print("theoretical_mean", theoretical_mean)


def evaluate_model_theoretical():
    alpha_q = pyro.param("alpha_q_concentration") / pyro.param("alpha_q_rate")
    beta_q = pyro.param("beta_q_concentration") / pyro.param("beta_q_rate")

    mean_theta = alpha_q / (alpha_q + beta_q)
    variance_theta = (alpha_q * beta_q) / (
        (alpha_q + beta_q) ** 2 * (alpha_q + beta_q + 1)
    )

    print(f"Estimated theoretical mean of theta: {mean_theta.item():.6f}")
    print(f"Estimated theoretical variance of theta: {variance_theta.item():.6f}")
    print(f"Estimated alpha: {alpha_q.item():.6f}")
    print(f"Estimated beta: {beta_q.item():.6f}")
    print(f"Data mean: {data.float().mean():.6f}")

    return mean_theta.item(), variance_theta.item()

