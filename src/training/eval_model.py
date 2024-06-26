import pyro


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

    return mean_theta.item(), variance_theta.item()
