import torch
import torch.nn.functional as F


def gaussian_elbo(x1, x2, z, sigma, mu, logvar):

    #
    # Problem 5b: Compute the evidence lower bound for the Gaussian VAE.
    #             Use the closed-form expression for the KL divergence from Problem 1.
    #

    # Calls in gaussina_vae.ipynb:
    # z, _, mu, logvar = f(x, epsilon)
    # loss_func(g(z),x,z,sigma,mu,logvar)
    # where
    # x1 = g(z)
    # x2 = x
    # mu = mu(q_phi(z|x))
    # sigma = sigma(p_theta(x|z))
    # logvar = log(sigma^2(q_phi(z|x)))

    reconstruction = 1 / (2 * sigma) * torch.linalg.norm(x2 - x1) ** 2
    divergence = 1 / 2 * torch.sum(torch.exp(logvar) + mu ** 2 - 1 - logvar)

    return reconstruction, divergence


def mc_gaussian_elbo(x1, x2, z, sigma, mu, logvar):

    #
    # Problem 5c: Compute the evidence lower bound for the Gaussian VAE.
    #             Use a (1-point) monte-carlo estimate of the KL divergence.
    #

    # Calls in gaussina_vae.ipynb:
    # z, _, mu, logvar = f(x, epsilon)
    # loss_func(g(z),x,z,sigma,mu,logvar)
    # where
    # x1 = g(z)
    # x2 = x
    # mu = mu(q_phi(z|x))
    # sigma = sigma(p_theta(x|z))
    # logvar = log(sigma^2(q_phi(z|x)))

    reconstruction = 1 / (2 * sigma) * torch.linalg.norm(x2 - x1) ** 2

    log_posterior = torch.distributions.MultivariateNormal(
        mu, torch.diag_embed(torch.exp(logvar))
    ).log_prob(z)
    d = z.shape[0]
    log_prior = torch.distributions.MultivariateNormal(
        torch.zeros(d), torch.eye(d)
    ).log_prob(z)

    divergence = torch.sum(log_posterior - log_prior)

    return reconstruction, divergence


def cross_entropy(x1, x2):
    return F.binary_cross_entropy_with_logits(x1, x2, reduction="sum") / x1.shape[0]


def discrete_output_elbo(x1, x2, z, logqzx):

    #
    # Problem 6b: Compute the evidence lower bound for a VAE with binary outputs.
    #             Use a (1-point) monte carlo estimate of the KL divergence.
    #

    # raise NotImplementedError
    reconstruction, divergence = 0, 0

    return reconstruction, divergence
