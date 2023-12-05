# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for MMDiff (https://github.com/Profluent-Internships/MMDiff):
# -------------------------------------------------------------------------------------------------------------------------------------

import math

import numpy as np
import torch
from beartype import beartype
from beartype.typing import Callable, List, Literal, Optional, Tuple, Union


@beartype
def betas_for_alpha_bar(num_timesteps: int, alpha_bar: Callable, max_beta: float = 0.999):
    """
    Create a beta schedule that discretizes the given `alpha_bar` function,
    which defines the cumulative product of `(1 - beta)` over time from `t = [0,1]`.
    :param num_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument `t` from `0` to `1` and
                      produces the cumulative product of `(1 - beta)` up to that
                      part of the diffusion process.
    :param max_beta: the maximum `beta` to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_timesteps):
        t1 = i / num_timesteps
        t2 = (i + 1) / num_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


@beartype
def get_named_beta_schedule(
    schedule_name: Literal["linear", "cosine", "sqrt"],
    num_timesteps: int,
    eps: float = 1e-4,
):
    """Get a pre-defined `beta` schedule for the given name.

    The `beta` schedule library consists of `beta` schedules which remain similar in the limit of
    `num_timesteps`. `Beta` schedules may be added, but should not be removed or changed once they
    are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # note: the linear schedule from Ho et al. 2020, extended to work for any number of diffusion timesteps
        scale = 1000 / num_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)

    if schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_timesteps, alpha_bar=lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        )

    if schedule_name == "sqrt":
        return betas_for_alpha_bar(num_timesteps, alpha_bar=lambda t: 1 - np.sqrt(t + eps))


@beartype
def extract_array_values(
    array: np.ndarray, timesteps: torch.Tensor, broadcast_shape: torch.Size
) -> torch.Tensor:
    """Extract values from a 1D NumPy array for a batch of indices.

    :param array: the 1D NumPy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of `K` dimensions with the batch
                            dimension equal to the length of `timesteps`.
    :return: a tensor of shape `[batch_size, 1, ...]` where the shape has `K` dims.
    """
    res = torch.as_tensor(array, device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class GaussianSequenceDiffusion:
    # adapted from: https://github.com/RosettaCommons/protein_generator
    def __init__(
        self,
        num_timesteps: int,
        noise_schedule: Literal["linear", "cosine", "sqrt"],
        sample_distribution: Literal["normal", "gmm"],
        sample_distribution_gmm_means: List[float] = [-1.0, 1.0],
        sample_distribution_gmm_variances: List[float] = [1.0, 1.0],
    ):
        assert len(sample_distribution_gmm_means) == len(
            sample_distribution_gmm_variances
        ), "Number of GMM means and variances provided must match."

        # use 64-bit floats for better numeric precision on timesteps
        self.betas = np.array(
            get_named_beta_schedule(noise_schedule, num_timesteps), dtype=np.float64
        )
        assert self.betas.ndim == 1, "`self.betas` must be one-dimensional."
        assert (self.betas > 0).all() and (
            self.betas <= 1
        ).all(), "`self.betas` must lie in expected numeric range of (0, 1]."

        self.num_timesteps = len(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (
            self.num_timesteps,
        ), "Cumulative product of (previous) `self.alphas` must match number of timesteps."

        # make calculations for posterior `q(x_{t-1} | x_t, x_0)`
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # note: log-calculation manually clipped because the posterior variance is `0` at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

        # make calculations for diffusion term `q(x_t | x_{t-1})` and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)

        # establish sample distribution arguments
        self.sample_distribution = sample_distribution
        self.sample_distribution_gmm_means = [
            float(mean) for mean in sample_distribution_gmm_means
        ]
        self.sample_distribution_gmm_variances = [
            float(variance) for variance in sample_distribution_gmm_variances
        ]

        if self.sample_distribution == "normal":
            self.noise_function = torch.randn_like
        else:
            self.noise_function = self.randn_mixture_like

    @beartype
    def q_mean_variance(
        self, x_start: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the distribution `q(x_t | x_0)`.

        :param x_start: the `[N x C x ...]` tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus `1`). Here, `0` means one step.
        :return: A tuple (mean, variance, log_variance), all of `x_start`'s shape.
        """
        mean = extract_array_values(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_array_values(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_array_values(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    @beartype
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        device: Optional[Union[torch.device, str]] = None,
        noise_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Diffuse the data for a given number of diffusion timesteps.

        In other words, sample from `q(x_t | x_0)`.
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus `1`). Here, `0` means one step.
        :param eps: if specified, the normal noise to apply.
        :param mask: if specified, the "fixed node" mask to use to leave "fixed" nodes without noise.
        :param device: if specified, the device on which to load the noise tensor.
        :param noise_scale: a noise scale factor for temperature-based sequence sampling.
        :return: A noisy version of `x_start`, along with its corresponding noise (i.e., epsilon).
        """

        # note: `self.noise_function` has previously been defined in `self.__init__()` depending on type of noise specified
        noise = eps if eps is not None else self.noise_function(x_start) * noise_scale
        if device is not None:
            noise = noise.to(device)

        assert (
            noise.shape == x_start.shape
        ), "Sampled noise must match the shape of the input `x_start`."
        x_sample = (
            extract_array_values(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_array_values(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        if mask is not None:
            x_sample[mask] = x_start[mask]

        return x_sample, noise

    @beartype
    def q_posterior_mean_variance(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the mean and variance of the diffusion posterior.

        In other words, derive `q(x_{t-1} | x_t, x_0)`.
        """
        assert x_start.shape == x_t.shape, "The inputs `x_start` and `x_t` must match shapes."

        posterior_mean = (
            extract_array_values(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_array_values(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_array_values(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_array_values(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        assert (
            len(posterior_mean)
            == len(posterior_variance)
            == len(posterior_log_variance_clipped)
            == len(x_start)
        ), "All distribution parameters must match the shape of the input `x_start`."
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @beartype
    def randn_mixture_like(
        self,
        tensor_like: torch.Tensor,
        num_gmms: int = 3,
        weights_normal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if weights_normal is not None:
            assert (
                len(weights_normal) == num_gmms
            ), "Number of GMMs must match number of distribution weights provided."
            mix = torch.distributions.Categorical(weights_normal)
        else:
            mix = torch.distributions.Categorical(
                torch.ones(len(self.sample_distribution_gmm_means))
            )
        comp = torch.distributions.Normal(
            torch.tensor(self.sample_distribution_gmm_means),
            torch.tensor(self.sample_distribution_gmm_variances),
        )
        # comp = torch.distributions.Normal([-3, 3], [1, 1])  # note: an example of a two-dimensional GMM
        # comp = torch.distributions.Normal([-3, 0, 3], [1, 1, 1])  # note: an example of a three-dimensional GMM
        gmm = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)
        return torch.tensor(
            [gmm.sample() for _ in range(np.prod(tensor_like.shape))], device=tensor_like.device
        ).reshape(tensor_like.shape)
