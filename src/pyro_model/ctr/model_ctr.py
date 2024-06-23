import pyro
import pandas as pd
import pyro.distributions as dist
import torch


def model_ctr(data: torch.tensor):
    alpha = pyro.sample("alpha", dist.Gamma(1, 1))
    beta = pyro.sample("beta", dist.Gamma(1, 1))

    with pyro.plate('data', len(data)):
        theta = pyro.sample("theta", dist.Beta(alpha, beta))
        pyro.sample('obs', dist.Bernoulli(theta), obs=data)


def guide_ctr(data: torch.tensor):
    alpha_q_concentration = pyro.param(
        "alpha_q_concentration", torch.tensor(1.0), constraint=dist.constraints.positive
    )
    alpha_q_rate = pyro.param(
        "alpha_q_rate", torch.tensor(1.0), constraint=dist.constraints.positive
    )
    beta_q_concentration = pyro.param(
        "beta_q_concentration", torch.tensor(1.0), constraint=dist.constraints.positive
    )
    beta_q_rate = pyro.param(
        "beta_q_rate", torch.tensor(1.0), constraint=dist.constraints.positive
    )

    alpha_q = pyro.sample("alpha", dist.Gamma(alpha_q_concentration, alpha_q_rate))
    beta_q = pyro.sample("beta", dist.Gamma(beta_q_concentration, beta_q_rate))

    with pyro.plate('data', len(data)):
        pyro.sample("theta", dist.Beta(alpha_q, beta_q))
