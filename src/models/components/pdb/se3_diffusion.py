# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from se3_diffusion (https://github.com/jasonkyuyim/se3_diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import datetime
import math
import os
import re
import tempfile
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from beartype.typing import Any, Dict, Optional, Tuple, Union
from jaxtyping import Bool, Float, jaxtyped
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation

import src.models.components.pdb.analysis_utils as au
from src.data.components.pdb import all_atom
from src.data.components.pdb import data_utils as du
from src.data.components.pdb import rigid_utils as ru
from src.data.components.pdb.complex_constants import NUM_NA_TORSIONS, NUM_PROT_TORSIONS
from src.data.components.pdb.data_transforms import convert_na_aatype6_to_aatype9
from src.data.components.pdb.pdb_na_dataset import (
    BATCH_TYPE,
    NUM_PROTEIN_NA_ONEHOT_AATYPE_CLASSES,
)
from src.models import detach_tensor_to_np
from src.models.components.pdb.embedders import RelPosEncoder
from src.models.components.pdb.framediff import FrameDiff
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


TIMESTEP_TYPE = Union[torch.Tensor, np.ndarray, np.float64, np.int64, float, int]
FLOAT_TIMESTEP_TYPE = Float[torch.Tensor, "batch_size"]  # noqa: F722
BATCH_MASK_TENSOR_TYPE = Bool[torch.Tensor, "batch_size"]  # noqa: F722
NODE_MASK_TENSOR_TYPE = Union[Float[torch.Tensor, "... num_nodes"], np.ndarray]  # noqa: F722
NODE_UPDATE_MASK_TENSOR_TYPE = Union[
    Float[torch.Tensor, "... num_nodes 1"], np.ndarray  # noqa: F722
]
QUATERNION_TENSOR_TYPE = Union[Float[torch.Tensor, "... num_nodes 4"], np.ndarray]  # noqa: F722
ROTATION_TENSOR_TYPE = Union[Float[torch.Tensor, "... 3 3"], np.ndarray]  # noqa: F722
COORDINATES_TENSOR_TYPE = Union[Float[torch.Tensor, "... num_nodes 3"], np.ndarray]  # noqa: F722


@beartype
def igso3_expansion(omega, eps, L: int = 1000, use_torch: bool = False):
    """Truncated sum of IGSO(3) distribution.

    This function approximates the power series in equation 5 of
    "DENOISING DIFFUSION PROBABILISTIC MODELS ON SO(3) FOR ROTATIONAL
    ALIGNMENT"
    Leach et al. 2022

    This expression diverges from the expression in Leach in that here, eps =
    sqrt(2) * eps_leach, if eps_leach were the scale parameter of the IGSO(3).

    With this reparameterization, IGSO(3) agrees with the Brownian motion on
    SO(3) with t=eps^2.

    Args:
        omega: rotation of Euler vector (i.e. the angle of rotation)
        eps: std of IGSO(3).
        L: Truncation level
        use_torch: set to `True` to use PyTorch Tensors; otherwise, use NumPy arrays.
    """

    lib = torch if use_torch else np
    ls = lib.arange(L)
    if use_torch:
        ls = ls.to(omega.device)
    if len(omega.shape) == 2:
        # Note: Used during predicted score calculation.
        ls = ls[None, None]  # [1, 1, L]
        omega = omega[..., None]  # [num_batch, num_res, 1]
        eps = eps[..., None]
    elif len(omega.shape) == 1:
        # Note: Used during cache computation.
        ls = ls[None]  # [1, L]
        omega = omega[..., None]  # [num_batch, 1]
    else:
        raise ValueError("`omega` must be 1D or 2D.")
    p = (
        (2 * ls + 1)
        * lib.exp(-ls * (ls + 1) * eps**2 / 2)
        * lib.sin(omega * (ls + 1 / 2))
        / lib.sin(omega / 2)
    )
    if use_torch:
        return p.sum(dim=-1)
    else:
        return p.sum(axis=-1)


@beartype
def density(expansion, omega, marginal: bool = True):
    """IGSO(3) density.

    Args:
        expansion: truncated approximation of the power series in the IGSO(3)
        density.
        omega: length of an Euler vector (i.e. angle of rotation)
        marginal: set true to give marginal density over the angle of rotation,
            otherwise include normalization to give density on SO(3) or a
            rotation with angle omega.
    """
    if marginal:
        # if marginal, density over [0, pi], else over SO(3)
        return expansion * (1 - np.cos(omega)) / np.pi
    else:
        # the constant factor doesn't affect any actual calculations though
        return expansion / 8 / np.pi**2


@beartype
def score(exp, omega, eps, L: int = 1000, use_torch: bool = False):  # score of density over SO(3)
    """Score uses the quotient rule to compute the scaling factor for the score of the IGSO(3)
    density.

    This function is used within the Diffuser class to when computing the score
    as an element of the tangent space of SO(3).

    This uses the quotient rule of calculus, and take the derivative of the
    log:
        d hi(x)/lo(x) = (lo(x) d hi(x)/dx - hi(x) d lo(x)/dx) / lo(x)^2
    and
        d log expansion(x) / dx = (d expansion(x)/ dx) / expansion(x)

    Args:
        exp: truncated expansion of the power series in the IGSO(3) density
        omega: length of an Euler vector (i.e. angle of rotation)
        eps: scale parameter for IGSO(3) -- as in expansion() this scaling
            differ from that in Leach by a factor of sqrt(2).
        L: truncation level
        use_torch: set to `True` to use PyTorch Tensors; otherwise, use NumPy arrays.

    Returns:
        The d/d omega log IGSO3(omega; eps)/(1-cos(omega))
    """

    lib = torch if use_torch else np
    ls = lib.arange(L)
    if use_torch:
        ls = ls.to(omega.device)
    ls = ls[None]
    if len(omega.shape) == 2:
        ls = ls[None]
    elif len(omega.shape) > 2:
        raise ValueError("`omega` must be 1D or 2D.")
    omega = omega[..., None]
    eps = eps[..., None]
    hi = lib.sin(omega * (ls + 1 / 2))
    dhi = (ls + 1 / 2) * lib.cos(omega * (ls + 1 / 2))
    lo = lib.sin(omega / 2)
    dlo = 1 / 2 * lib.cos(omega / 2)
    dSigma = (
        (2 * ls + 1) * lib.exp(-ls * (ls + 1) * eps**2 / 2) * (lo * dhi - hi * dlo) / lo**2
    )
    if use_torch:
        dSigma = dSigma.sum(dim=-1)
    else:
        dSigma = dSigma.sum(axis=-1)
    return dSigma / (exp + 1e-4)


@beartype
def _extract_trans_rots(rigid: ru.Rigid) -> Tuple[np.ndarray, np.ndarray]:
    rot = rigid.get_rots().get_rot_mats().cpu().numpy()
    rot_shape = rot.shape
    num_rots = np.cumprod(rot_shape[:-2])[-1]
    rot = rot.reshape((num_rots, 3, 3))
    rot = Rotation.from_matrix(rot).as_rotvec().reshape(rot_shape[:-2] + (3,))
    tran = rigid.get_trans().cpu().numpy()
    return tran, rot


@beartype
def _assemble_rigid(rotvec: np.ndarray, trans: np.ndarray) -> ru.Rigid:
    rotvec_shape = rotvec.shape
    num_rotvecs = np.cumprod(rotvec_shape[:-1])[-1]
    rotvec = rotvec.reshape((num_rotvecs, 3))
    rotmat = Rotation.from_rotvec(rotvec).as_matrix().reshape(rotvec_shape[:-1] + (3, 3))
    return ru.Rigid(rots=ru.Rotation(rot_mats=torch.Tensor(rotmat)), trans=torch.tensor(trans))


@beartype
def compose_rotvec(r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """Compose two rotation euler vectors."""
    R1 = rotvec_to_matrix(r1)
    R2 = rotvec_to_matrix(r2)
    cR = np.einsum("...ij,...jk->...ik", R1, R2)
    return matrix_to_rotvec(cR)


@beartype
def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(rotvec).as_matrix()


@beartype
def matrix_to_rotvec(mat: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(mat).as_rotvec()


@beartype
def rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(rotvec).as_quat()


@jaxtyped
@beartype
def quat_to_rotvec(
    quat: QUATERNION_TENSOR_TYPE, mask: Optional[NODE_MASK_TENSOR_TYPE] = None, eps: float = 1e-6
) -> COORDINATES_TENSOR_TYPE:
    # note: w > 0 to ensure 0 <= angle <= pi
    flip = (quat[..., :1] < 0).float()
    quat = (-1 * quat) * flip + (1 - flip) * quat

    if mask is not None:
        # avoid `torch.atan2()` attempting to backpropagate masked nodes' "missing" values
        angle, angle_mask = quat[..., 0].detach(), mask.bool()
        angle[angle_mask] = 2 * torch.atan2(
            torch.linalg.norm(quat[..., 1:][angle_mask], dim=-1), quat[..., 0][angle_mask]
        )
    else:
        angle = 2 * torch.atan2(torch.linalg.norm(quat[..., 1:], dim=-1), quat[..., 0])

    angle2 = angle * angle
    small_angle_scales = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880
    large_angle_scales = angle / torch.sin(angle / 2 + eps)

    small_angles = (angle <= 1e-3).float()
    rot_vec_scale = small_angle_scales * small_angles + (1 - small_angles) * large_angle_scales
    rot_vec = rot_vec_scale[..., None] * quat[..., 1:]
    return rot_vec


@beartype
def calculate_timestep_embedding(
    timesteps: torch.Tensor, embedding_dim: int, max_positions: int = 10000
) -> torch.Tensor:
    # Adapted from: https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class SO3Diffusion:
    def __init__(self, diffusion_cfg: DictConfig):
        self.schedule = diffusion_cfg.schedule

        self.min_sigma = diffusion_cfg.min_sigma
        self.max_sigma = diffusion_cfg.max_sigma

        self.num_sigma = diffusion_cfg.num_sigma
        self.use_cached_score = diffusion_cfg.use_cached_score

        # discretize `omegas` for calculating CDFs; skip `omega=0`
        self.discrete_omega = np.linspace(0, np.pi, diffusion_cfg.num_omega + 1)[1:]

        # precompute IGSO3 values
        def replace_period(x):
            return str(x).replace(".", "_")

        cache_dir = os.path.join(
            diffusion_cfg.cache_dir,
            f"eps_{diffusion_cfg.num_sigma}_omega_{diffusion_cfg.num_omega}_min_sigma_{replace_period(diffusion_cfg.min_sigma)}_max_sigma_{replace_period(diffusion_cfg.max_sigma)}_schedule_{diffusion_cfg.schedule}",
        )

        # if cache directory does not exist, create it
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        pdf_cache = os.path.join(cache_dir, "pdf_vals.npy")
        cdf_cache = os.path.join(cache_dir, "cdf_vals.npy")
        score_norms_cache = os.path.join(cache_dir, "score_norms.npy")

        if (
            os.path.exists(pdf_cache)
            and os.path.exists(cdf_cache)
            and os.path.exists(score_norms_cache)
        ):
            log.info(f"Using cached IGSO3 in {cache_dir}")
            self._pdf = np.load(pdf_cache)
            self._cdf = np.load(cdf_cache)
            self._score_norms = np.load(score_norms_cache)
        else:
            log.info(f"Computing IGSO3. Saving in {cache_dir}")
            # compute the expansion of the power series
            exp_vals = np.asarray(
                [igso3_expansion(self.discrete_omega, sigma) for sigma in self.discrete_sigma]
            )
            # compute the PDF and CDF values for the marginal distribution of the angle of rotation (which is needed for sampling)
            self._pdf = np.asarray(
                [density(x, self.discrete_omega, marginal=True) for x in exp_vals]
            )
            self._cdf = np.asarray(
                [pdf.cumsum() / diffusion_cfg.num_omega * np.pi for pdf in self._pdf]
            )

            # compute the norms of the scores
            # note: this is used to scale the rotation axis when computing the score as a vector
            self._score_norms = np.asarray(
                [
                    score(exp_vals[i], self.discrete_omega, x)
                    for i, x in enumerate(self.discrete_sigma)
                ]
            )

            # cache the precomputed values
            np.save(pdf_cache, self._pdf)
            np.save(cdf_cache, self._cdf)
            np.save(score_norms_cache, self._score_norms)

        self._score_scaling = np.sqrt(
            np.abs(
                np.sum(self._score_norms**2 * self._pdf, axis=-1) / np.sum(self._pdf, axis=-1)
            )
        ) / np.sqrt(3)

    @property
    def discrete_sigma(self) -> TIMESTEP_TYPE:
        return self.sigma(np.linspace(0.0, 1.0, self.num_sigma))

    @jaxtyped
    @beartype
    def sigma_idx(self, sigma: TIMESTEP_TYPE) -> TIMESTEP_TYPE:
        """Calculates the index for discretized sigma during IGSO(3) initialization."""
        return np.digitize(sigma, self.discrete_sigma) - 1

    @jaxtyped
    @beartype
    def sigma(self, t: TIMESTEP_TYPE) -> TIMESTEP_TYPE:
        r"""Extract \sigma(t) corresponding to chosen sigma schedule."""
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError(f"Invalid t={t}")
        if self.schedule == "logarithmic":
            return np.log(t * np.exp(self.max_sigma) + (1 - t) * np.exp(self.min_sigma))
        else:
            raise ValueError(f"Unrecognize schedule {self.schedule}")

    @jaxtyped
    @beartype
    def diffusion_coef(self, t: TIMESTEP_TYPE) -> TIMESTEP_TYPE:
        """Compute diffusion coefficient (g_t)."""
        if self.schedule == "logarithmic":
            g_t = np.sqrt(
                2
                * (np.exp(self.max_sigma) - np.exp(self.min_sigma))
                * self.sigma(t)
                / np.exp(self.sigma(t))
            )
        else:
            raise ValueError(f"Unrecognize schedule {self.schedule}")
        return g_t

    @jaxtyped
    @beartype
    def t_to_idx(self, t: TIMESTEP_TYPE) -> TIMESTEP_TYPE:
        """Helper function to go from time t to corresponding sigma_idx."""
        return self.sigma_idx(self.sigma(t))

    @jaxtyped
    @beartype
    def sample_igso3(self, t: TIMESTEP_TYPE, num_samples: Union[int, np.int64] = 1) -> np.ndarray:
        """Uses the inverse cdf to sample an angle of rotation from IGSO(3).

        Args:
            t: continuous time in [0, 1].
            num_samples: number of samples to draw.

        Returns:
            [num_samples] angles of rotation.
        """
        if not np.isscalar(t):
            raise ValueError(f"{t} must be a scalar.")
        x = np.random.rand(num_samples)
        return np.interp(x, self._cdf[self.t_to_idx(t)], self.discrete_omega)

    @jaxtyped
    @beartype
    def sample(self, t: TIMESTEP_TYPE, num_samples: Union[int, np.int64] = 1) -> np.ndarray:
        """Generates rotation vector(s) from IGSO(3).

        Args:
            t: continuous time in [0, 1].
            n_sample: number of samples to generate.

        Returns:
            [num_samples, 3] axis-angle rotation vectors sampled from IGSO(3).
        """
        x = np.random.randn(num_samples, 3)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        return x * self.sample_igso3(t, num_samples=num_samples)[:, None]

    @beartype
    def sample_ref(self, num_samples: Union[int, np.int64] = 1) -> np.ndarray:
        return self.sample(1, num_samples=num_samples)

    @jaxtyped
    @beartype
    def score(
        self,
        vec: COORDINATES_TENSOR_TYPE,
        t: TIMESTEP_TYPE,
    ) -> COORDINATES_TENSOR_TYPE:
        """Computes the score of IGSO(3) density as a rotation vector.

        Args:
            vec: [..., 3] array of axis-angle rotation vectors.
            t: continuous time in [0, 1].

        Returns:
            [..., 3] score vector in the direction of the sampled vector with
            magnitude given by _score_norms.
        """
        if not np.isscalar(t):
            raise ValueError(f"{t} must be a scalar.")
        torch_score = self.torch_score(torch.tensor(vec), torch.tensor(t)[None])
        return torch_score.numpy()

    @jaxtyped
    @beartype
    def torch_score(
        self, vec: COORDINATES_TENSOR_TYPE, t: TIMESTEP_TYPE, eps: float = 1e-6
    ) -> COORDINATES_TENSOR_TYPE:
        """Computes the score of IGSO(3) density as a rotation vector.

        Same as score function but uses pytorch and performs a look-up.

        Args:
            vec: [..., 3] array of axis-angle rotation vectors.
            t: continuous time in [0, 1].

        Returns:
            [..., 3] score vector in the direction of the sampled vector with
            magnitude given by _score_norms.
        """
        omega = torch.linalg.norm(vec, dim=-1) + eps
        if self.use_cached_score:
            score_norms_t = self._score_norms[self.t_to_idx(detach_tensor_to_np(t))]
            score_norms_t = torch.tensor(score_norms_t, device=vec.device)
            omega_idx = torch.bucketize(
                omega, torch.tensor(self.discrete_omega[:-1], device=vec.device)
            )
            omega_scores_t = torch.gather(score_norms_t, 1, omega_idx)
        else:
            sigma = self.discrete_sigma[self.t_to_idx(detach_tensor_to_np(t))]
            sigma = torch.tensor(sigma, device=vec.device, dtype=vec.dtype)
            omega_vals = igso3_expansion(omega, sigma[:, None], use_torch=True)
            omega_scores_t = score(omega_vals, omega, sigma[:, None], use_torch=True)
        return omega_scores_t[..., None] * vec / (omega[..., None] + eps)

    @jaxtyped
    @beartype
    def score_scaling(self, t: TIMESTEP_TYPE) -> TIMESTEP_TYPE:
        """Calculates scaling used for scores during training."""
        return self._score_scaling[self.t_to_idx(t)]

    @jaxtyped
    @beartype
    def forward_marginal(
        self, rots_0: np.ndarray, t: TIMESTEP_TYPE
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Samples from the forward diffusion process at time index t.

        Args:
            rots_0: [..., 3] initial rotations.
            t: continuous time in [0, 1].

        Returns:
            rot_t: [..., 3] noised rotation vectors.
            rot_score: [..., 3] score of rot_t as a rotation vector.
        """
        num_samples = np.cumprod(rots_0.shape[:-1])[-1]
        sampled_rots = self.sample(t, num_samples=num_samples)
        rot_score = self.score(sampled_rots, t).reshape(rots_0.shape)
        rot_t = compose_rotvec(sampled_rots, rots_0).reshape(rots_0.shape)
        return rot_t, rot_score

    @jaxtyped
    @beartype
    def reverse(
        self,
        rot_t: ROTATION_TENSOR_TYPE,
        score_t: ROTATION_TENSOR_TYPE,
        t: float,
        dt: float,
        mask: Optional[NODE_MASK_TENSOR_TYPE] = None,
        noise_scale: float = 1.0,
    ) -> np.ndarray:
        """Simulates the reverse SDE for 1 step using the Geodesic random walk.

        Args:
            rot_t: [..., 3, 3] current rotations at time t.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            add_noise: set False to set diffusion coefficient to 0.
            mask: True indicates which residues to diffuse.
            noise_scale: Scale factor for noise.

        Returns:
            [..., 3] rotation vector at next step.
        """
        if not np.isscalar(t):
            raise ValueError(f"{t} must be a scalar.")

        g_t = self.diffusion_coef(t)
        z = noise_scale * np.random.normal(size=score_t.shape)
        perturb = (g_t**2) * score_t * dt + g_t * np.sqrt(dt) * z

        if mask is not None:
            perturb = perturb * mask[..., None]
        num_samples = np.cumprod(rot_t.shape[:-1])[-1]

        # left-multiply
        rot_t_1 = compose_rotvec(
            perturb.reshape(num_samples, 3),
            rot_t.reshape(num_samples, 3),
        ).reshape(rot_t.shape)
        return rot_t_1


class R3Diffusion:
    """VP-SDE diffusion class for translations."""

    def __init__(self, diffusion_cfg: DictConfig):
        """
        Args:
            min_b: starting value in variance schedule.
            max_b: ending value in variance schedule.
        """
        self.diffusion_cfg = diffusion_cfg
        self.min_b = diffusion_cfg.min_b
        self.max_b = diffusion_cfg.max_b

    @jaxtyped
    @beartype
    def _scale(self, x: COORDINATES_TENSOR_TYPE) -> COORDINATES_TENSOR_TYPE:
        return x * self.diffusion_cfg.coordinate_scaling

    @jaxtyped
    @beartype
    def _unscale(self, x: COORDINATES_TENSOR_TYPE) -> COORDINATES_TENSOR_TYPE:
        return x / self.diffusion_cfg.coordinate_scaling

    @jaxtyped
    @beartype
    def b_t(self, t: TIMESTEP_TYPE) -> TIMESTEP_TYPE:
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError(f"Invalid t={t}")
        return self.min_b + t * (self.max_b - self.min_b)

    @jaxtyped
    @beartype
    def diffusion_coef(self, t: TIMESTEP_TYPE) -> TIMESTEP_TYPE:
        """Time-dependent diffusion coefficient."""
        return np.sqrt(self.b_t(t))

    @jaxtyped
    @beartype
    def drift_coef(self, x: COORDINATES_TENSOR_TYPE, t: TIMESTEP_TYPE) -> COORDINATES_TENSOR_TYPE:
        """Time-dependent drift coefficient."""
        return -1 / 2 * self.b_t(t) * x

    @beartype
    def sample_ref(self, num_samples: Union[int, np.int64] = 1) -> np.ndarray:
        return np.random.normal(size=(num_samples, 3))

    @jaxtyped
    @beartype
    def marginal_b_t(self, t: TIMESTEP_TYPE) -> TIMESTEP_TYPE:
        return t * self.min_b + (1 / 2) * (t**2) * (self.max_b - self.min_b)

    @jaxtyped
    @beartype
    def calc_trans_0(
        self,
        score_t: COORDINATES_TENSOR_TYPE,
        x_t: COORDINATES_TENSOR_TYPE,
        t: TIMESTEP_TYPE,
        use_torch: bool = True,
    ) -> COORDINATES_TENSOR_TYPE:
        beta_t = self.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        exp_fn = torch.exp if use_torch else np.exp
        cond_var = 1 - exp_fn(-beta_t)
        return (score_t * cond_var + x_t) / exp_fn(-1 / 2 * beta_t)

    @jaxtyped
    @beartype
    def forward(
        self, x_t_1: COORDINATES_TENSOR_TYPE, t: TIMESTEP_TYPE, num_t: Union[int, np.int64]
    ) -> COORDINATES_TENSOR_TYPE:
        """Samples marginal p(x(t) | x(t-1)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        if not np.isscalar(t):
            raise ValueError(f"{t} must be a scalar.")
        x_t_1 = self._scale(x_t_1)
        b_t = torch.tensor(self.marginal_b_t(t) / num_t, device=x_t_1.device)
        z_t_1 = torch.tensor(np.random.normal(size=x_t_1.shape), device=x_t_1.device)
        x_t = torch.sqrt(1 - b_t) * x_t_1 + torch.sqrt(b_t) * z_t_1
        return x_t

    @jaxtyped
    @beartype
    def distribution(
        self,
        x_t: COORDINATES_TENSOR_TYPE,
        score_t: COORDINATES_TENSOR_TYPE,
        t: TIMESTEP_TYPE,
        dt: TIMESTEP_TYPE,
        mask: Optional[NODE_MASK_TENSOR_TYPE] = None,
    ) -> Tuple[COORDINATES_TENSOR_TYPE, TIMESTEP_TYPE]:
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        std = g_t * np.sqrt(dt)
        mu = x_t - (f_t - g_t**2 * score_t) * dt
        if mask is not None:
            mu = mu * mask[..., None]
        return mu, std

    @jaxtyped
    @beartype
    def forward_marginal(
        self, x_0: COORDINATES_TENSOR_TYPE, t: TIMESTEP_TYPE
    ) -> Tuple[COORDINATES_TENSOR_TYPE, COORDINATES_TENSOR_TYPE]:
        """Samples marginal p(x(t) | x(0)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        if not np.isscalar(t):
            raise ValueError(f"{t} must be a scalar.")
        x_0 = self._scale(x_0)
        x_t = np.random.normal(
            loc=np.exp(-1 / 2 * self.marginal_b_t(t)) * x_0,
            scale=np.sqrt(1 - np.exp(-self.marginal_b_t(t))),
        )
        score_t = self.score(x_t, x_0, t)
        x_t = self._unscale(x_t)
        return x_t, score_t

    @jaxtyped
    @beartype
    def score_scaling(self, t: TIMESTEP_TYPE) -> TIMESTEP_TYPE:
        return 1 / np.sqrt(self.conditional_var(t))

    @jaxtyped
    @beartype
    def reverse(
        self,
        x_t: COORDINATES_TENSOR_TYPE,
        score_t: COORDINATES_TENSOR_TYPE,
        t: TIMESTEP_TYPE,
        dt: TIMESTEP_TYPE,
        mask: Optional[NODE_MASK_TENSOR_TYPE] = None,
        center: bool = True,
        noise_scale: float = 1.0,
    ) -> COORDINATES_TENSOR_TYPE:
        """Simulates the reverse SDE for 1 step.

        Args:
            x_t: [..., 3] current positions at time t in angstroms.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: True indicates which residues to diffuse.

        Returns:
            [..., 3] positions at next step t-1.
        """
        if not np.isscalar(t):
            raise ValueError(f"{t} must be a scalar.")
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        z = noise_scale * np.random.normal(size=score_t.shape)
        perturb = (f_t - g_t**2 * score_t) * dt + g_t * np.sqrt(dt) * z

        if mask is not None:
            perturb = perturb * mask[..., None]
        else:
            mask = np.ones(x_t.shape[:-1])
        x_t_1 = x_t - perturb
        if center:
            com = np.sum(x_t_1, axis=-2) / np.sum(mask, axis=-1)[..., None]
            x_t_1 -= com[..., None, :]
        x_t_1 = self._unscale(x_t_1)
        return x_t_1

    @jaxtyped
    @beartype
    def conditional_var(self, t: TIMESTEP_TYPE, use_torch: bool = False) -> TIMESTEP_TYPE:
        """Conditional variance of p(xt|x0).

        Var[x_t|x_0] = conditional_var(t)*I
        """
        if use_torch:
            return 1 - torch.exp(-self.marginal_b_t(t))
        return 1 - np.exp(-self.marginal_b_t(t))

    @jaxtyped
    @beartype
    def score(
        self,
        x_t: COORDINATES_TENSOR_TYPE,
        x_0: COORDINATES_TENSOR_TYPE,
        t: TIMESTEP_TYPE,
        use_torch: bool = False,
        scale: bool = False,
    ) -> COORDINATES_TENSOR_TYPE:
        if use_torch:
            exp_fn = torch.exp
        else:
            exp_fn = np.exp
        if scale:
            x_t = self._scale(x_t)
            x_0 = self._scale(x_0)
        return -(x_t - exp_fn(-1 / 2 * self.marginal_b_t(t)) * x_0) / self.conditional_var(
            t, use_torch=use_torch
        )


class SE3Diffusion:
    def __init__(self, diffusion_cfg: DictConfig, **kwargs):
        self.diffusion_cfg = diffusion_cfg

        self.diffuse_rotations = diffusion_cfg.diffuse_rotations
        self.so3_diffusion = SO3Diffusion(diffusion_cfg.so3)

        self.diffuse_translations = diffusion_cfg.diffuse_translations
        self.r3_diffusion = R3Diffusion(diffusion_cfg.r3)

    @jaxtyped
    @beartype
    def forward_marginal(
        self,
        rigids_0: ru.Rigid,
        t: float,
        diffuse_mask: Optional[NODE_MASK_TENSOR_TYPE] = None,
        as_tensor_7: bool = True,
    ):
        """
        Args:
            rigids_0: [..., N] OpenFold `Rigid` objects
            t: continuous time in [0, 1].

        Returns:
            rigids_t: [..., N] noised `Rigid`. [..., N, 7] if `as_tensor_7` is `True`.
            trans_score: [..., N, 3] translation score
            rot_score: [..., N, 3] rotation score
            trans_score_norm: [...] translation score norm
            rot_score_norm: [...] rotation score norm
        """
        trans_0, rot_0 = _extract_trans_rots(rigids_0)

        if not self.diffuse_rotations:
            rot_t, rot_score, rot_score_scaling = (rot_0, np.zeros_like(rot_0), np.ones_like(t))
        else:
            rot_t, rot_score = self.so3_diffusion.forward_marginal(rot_0, t)
            rot_score_scaling = self.so3_diffusion.score_scaling(t)

        if not self.diffuse_translations:
            trans_t, trans_score, trans_score_scaling = (
                trans_0,
                np.zeros_like(trans_0),
                np.ones_like(t),
            )
        else:
            trans_t, trans_score = self.r3_diffusion.forward_marginal(trans_0, t)
            trans_score_scaling = self.r3_diffusion.score_scaling(t)

        if diffuse_mask is not None:
            rot_t = self._apply_mask(rot_t, rot_0, diffuse_mask[..., None])
            trans_t = self._apply_mask(trans_t, trans_0, diffuse_mask[..., None])

            trans_score = self._apply_mask(
                trans_score, np.zeros_like(trans_score), diffuse_mask[..., None]
            )
            rot_score = self._apply_mask(
                rot_score, np.zeros_like(rot_score), diffuse_mask[..., None]
            )
        rigids_t = _assemble_rigid(rot_t, trans_t)
        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()
        return {
            "rigids_t": rigids_t,
            "trans_score": trans_score,
            "rot_score": rot_score,
            "trans_score_scaling": trans_score_scaling,
            "rot_score_scaling": rot_score_scaling,
        }

    @jaxtyped
    @beartype
    def calc_trans_0(
        self,
        trans_score: COORDINATES_TENSOR_TYPE,
        trans_t: COORDINATES_TENSOR_TYPE,
        t: TIMESTEP_TYPE,
    ) -> COORDINATES_TENSOR_TYPE:
        return self.r3_diffusion.calc_trans_0(trans_score, trans_t, t)

    def calc_trans_score(
        self,
        trans_t: COORDINATES_TENSOR_TYPE,
        trans_0: COORDINATES_TENSOR_TYPE,
        t: FLOAT_TIMESTEP_TYPE,
        use_torch: bool = False,
        scale: bool = True,
    ) -> COORDINATES_TENSOR_TYPE:
        return self.r3_diffusion.score(trans_t, trans_0, t, use_torch=use_torch, scale=scale)

    @jaxtyped
    @beartype
    def calc_rot_score(
        self,
        rots_t: ru.Rotation,
        rots_0: ru.Rotation,
        t: FLOAT_TIMESTEP_TYPE,
        mask: Optional[NODE_MASK_TENSOR_TYPE] = None,
    ) -> COORDINATES_TENSOR_TYPE:
        rots_0_inv = rots_0.invert(mask=mask)
        quats_0_inv = rots_0_inv.get_quats()
        quats_t = rots_t.get_quats()
        quats_0t = ru.quat_multiply(quats_0_inv, quats_t)
        rotvec_0t = quat_to_rotvec(quats_0t, mask=mask)
        return self.so3_diffusion.torch_score(rotvec_0t, t)

    @jaxtyped
    @beartype
    def _apply_mask(
        self,
        value_diff: Union[Float[torch.Tensor, "... num_dims"], np.ndarray],  # noqa: F722
        value: Union[Float[torch.Tensor, "... num_dims"], np.ndarray],  # noqa: F722
        diff_mask: NODE_UPDATE_MASK_TENSOR_TYPE,
    ) -> Union[Float[torch.Tensor, "batch_size num_nodes 2"], np.ndarray]:  # noqa: F722
        return diff_mask * value_diff + (1 - diff_mask) * value

    def trans_parameters(self, trans_t, score_t, t, dt, mask):
        return self.r3_diffusion.distribution(trans_t, score_t, t, dt, mask=mask)

    @jaxtyped
    @beartype
    def score(
        self, rigid_0: ru.Rigid, rigid_t: ru.Rigid, t: TIMESTEP_TYPE
    ) -> Tuple[COORDINATES_TENSOR_TYPE, ROTATION_TENSOR_TYPE]:
        tran_0, rot_0 = _extract_trans_rots(rigid_0)
        tran_t, rot_t = _extract_trans_rots(rigid_t)

        if not self.diffuse_rotations:
            rot_score = np.zeros_like(rot_0)
        else:
            rot_score = self.so3_diffusion.score(rot_t, t)

        if not self.diffuse_translations:
            trans_score = np.zeros_like(tran_0)
        else:
            trans_score = self.r3_diffusion.score(tran_t, tran_0, t)

        return trans_score, rot_score

    def score_scaling(
        self, t: TIMESTEP_TYPE
    ) -> Tuple[ROTATION_TENSOR_TYPE, COORDINATES_TENSOR_TYPE]:
        rot_score_scaling = self.so3_diffusion.score_scaling(t)
        trans_score_scaling = self.r3_diffusion.score_scaling(t)
        return rot_score_scaling, trans_score_scaling

    @jaxtyped
    @beartype
    def reverse(
        self,
        rigid_t: ru.Rigid,
        rot_score: ROTATION_TENSOR_TYPE,
        trans_score: COORDINATES_TENSOR_TYPE,
        t: TIMESTEP_TYPE,
        dt: TIMESTEP_TYPE,
        diffuse_mask: Optional[NODE_MASK_TENSOR_TYPE] = None,
        center: bool = True,
        noise_scale: float = 1.0,
    ) -> ru.Rigid:
        """Reverse sampling function from (t) to (t-1).

        Args:
            rigid_t: [..., N] protein rigid objects at time t.
            rot_score: [..., N, 3] rotation score.
            trans_score: [..., N, 3] translation score.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: [..., N] which residues to update.
            center: true to set center of mass to zero after step

        Returns:
            rigid_t_1: [..., N] protein rigid objects at time t-1.
        """
        trans_t, rot_t = _extract_trans_rots(rigid_t)
        if diffuse_mask is not None:
            trans_t = trans_t * diffuse_mask[..., None]
            rot_t = rot_t * diffuse_mask[..., None]
        if not self.diffuse_rotations:
            rot_t_1 = rot_t
        else:
            rot_t_1 = self.so3_diffusion.reverse(
                rot_t=rot_t,
                score_t=rot_score,
                t=t,
                dt=dt,
                noise_scale=noise_scale,
            )
        if not self.diffuse_translations:
            trans_t_1 = trans_t
        else:
            trans_t_1 = self.r3_diffusion.reverse(
                x_t=trans_t,
                score_t=trans_score,
                t=t,
                dt=dt,
                center=center,
                noise_scale=noise_scale,
            )

        if diffuse_mask is not None:
            trans_t_1 = self._apply_mask(trans_t_1, trans_t, diffuse_mask[..., None])
            rot_t_1 = self._apply_mask(rot_t_1, rot_t, diffuse_mask[..., None])

        return _assemble_rigid(rot_t_1, trans_t_1)

    @jaxtyped
    @beartype
    def sample_ref(
        self,
        num_samples: Union[int, np.int64],
        impute: Optional[ru.Rigid] = None,
        diffuse_mask: Optional[NODE_MASK_TENSOR_TYPE] = None,
        as_tensor_7: bool = False,
    ) -> Dict[str, Float[torch.Tensor, "... num_nodes 7"]]:  # noqa: F722
        """Samples rigids from reference distribution.

        Args:
            num_samples: Number of samples.
            impute: Rigid objects to use as imputation values if either
                translations or rotations are not diffused.
        """
        if impute is not None:
            assert impute.shape[0] == num_samples
            trans_impute, rot_impute = _extract_trans_rots(impute)
            trans_impute = trans_impute.reshape((num_samples, 3))
            rot_impute = rot_impute.reshape((num_samples, 3))
            trans_impute = self.r3_diffusion._scale(trans_impute)

        if diffuse_mask is not None and impute is None:
            raise ValueError("Must provide imputation values.")

        if (not self.diffuse_rotations) and impute is None:
            raise ValueError("Must provide imputation values.")

        if (not self.diffuse_translations) and impute is None:
            raise ValueError("Must provide imputation values.")

        if self.diffuse_rotations:
            rot_ref = self.so3_diffusion.sample_ref(num_samples=num_samples)
        else:
            rot_ref = rot_impute

        if self.diffuse_translations:
            trans_ref = self.r3_diffusion.sample_ref(num_samples=num_samples)
        else:
            trans_ref = trans_impute

        if diffuse_mask is not None:
            rot_ref = self._apply_mask(rot_ref, rot_impute, diffuse_mask[..., None])
            trans_ref = self._apply_mask(trans_ref, trans_impute, diffuse_mask[..., None])
        trans_ref = self.r3_diffusion._unscale(trans_ref)
        rigids_t = _assemble_rigid(rot_ref, trans_ref)
        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()
        return {"rigids_t": rigids_t}


class ScoreEmbedding(nn.Module):
    def __init__(self, model_cfg: DictConfig, diffusion_cfg: DictConfig):
        super().__init__()
        self.model_cfg = model_cfg
        self.diffusion_cfg = diffusion_cfg
        self.embedding_cfg = model_cfg.embedding

        # prepare timestep embedding
        index_embedding_size = self.embedding_cfg.index_embedding_size
        t_embedding_size = index_embedding_size
        node_embedding_dim = t_embedding_size + 1
        edge_embedding_dim = (t_embedding_size + 1) * 2

        # craft relative sequence position embedding
        self.relpos_embedder = RelPosEncoder(
            embedding_size=self.embedding_cfg.index_embedding_size,
            max_relative_idx=self.embedding_cfg.max_relative_idx,
            max_relative_chain=self.embedding_cfg.max_relative_chain,
            use_chain_relative=self.embedding_cfg.use_chain_relative,
        )
        edge_embedding_dim += self.embedding_cfg.index_embedding_size

        # accommodate the ability to condition on per-residue molecule types
        if self.embedding_cfg.embed_molecule_type_conditioning:
            self.molecule_type_embedder = nn.Sequential(
                nn.Linear(
                    self.embedding_cfg.molecule_type_embedding_size,
                    self.embedding_cfg.molecule_type_embedded_size,
                ),
                nn.ReLU(),
                nn.Linear(
                    self.embedding_cfg.molecule_type_embedded_size,
                    self.embedding_cfg.molecule_type_embedded_size,
                ),
            )
            node_embedding_dim += self.embedding_cfg.molecule_type_embedded_size

        # add the ability to learn sequence embeddings
        self.diffuse_sequence = (
            hasattr(diffusion_cfg, "diffuse_sequence") and diffusion_cfg.diffuse_sequence
        )
        if self.diffuse_sequence:
            node_embedding_dim += self.model_cfg.node_hidden_dim
            self.node_type_embedder = nn.Embedding(
                NUM_PROTEIN_NA_ONEHOT_AATYPE_CLASSES, self.model_cfg.node_hidden_dim
            )

        if self.embedding_cfg.embed_self_conditioning and self.diffuse_sequence:
            node_embedding_dim += NUM_PROTEIN_NA_ONEHOT_AATYPE_CLASSES
        node_hidden_dim = self.model_cfg.node_hidden_dim
        self.node_embedder = nn.Sequential(
            nn.Linear(node_embedding_dim, node_hidden_dim),
            nn.ReLU(),
            nn.Linear(node_hidden_dim, node_hidden_dim),
            nn.ReLU(),
            nn.Linear(node_hidden_dim, node_hidden_dim),
            nn.LayerNorm(node_hidden_dim),
        )

        # incorporate the ability to condition on a diffused input sequence
        if self.diffuse_sequence:
            edge_embedding_dim += self.model_cfg.edge_hidden_dim
            self.row_onehot_node_type_embedder = nn.Embedding(
                NUM_PROTEIN_NA_ONEHOT_AATYPE_CLASSES, self.model_cfg.edge_hidden_dim
            )
            self.col_onehot_node_type_embedder = nn.Embedding(
                NUM_PROTEIN_NA_ONEHOT_AATYPE_CLASSES, self.model_cfg.edge_hidden_dim
            )

        if self.embedding_cfg.embed_self_conditioning:
            edge_embedding_dim += self.embedding_cfg.num_bins
        edge_hidden_dim = self.model_cfg.edge_hidden_dim
        self.edge_embedder = nn.Sequential(
            nn.Linear(edge_embedding_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.LayerNorm(edge_hidden_dim),
        )

        self.timestep_embedder = partial(
            calculate_timestep_embedding, embedding_dim=self.embedding_cfg.index_embedding_size
        )

    @jaxtyped
    @beartype
    def cross_concat(
        self,
        node_feats: Float[
            torch.Tensor, "batch_size num_nodes num_node_hidden_channels"  # noqa: F722
        ],
    ) -> Float[torch.Tensor, "batch_size num_edges num_edge_hidden_channels"]:  # noqa: F722
        batch_size, num_nodes = node_feats.shape[0], node_feats.shape[1]
        return (
            torch.cat(
                [
                    torch.tile(node_feats[:, :, None, :], (1, 1, num_nodes, 1)),
                    torch.tile(node_feats[:, None, :, :], (1, num_nodes, 1, 1)),
                ],
                dim=-1,
            )
            .float()
            .reshape([batch_size, num_nodes**2, -1])
        )

    @jaxtyped
    @beartype
    def forward(
        self, batch: BATCH_TYPE, fixed_node_eps: float = 1e-5
    ) -> Tuple[
        Float[torch.Tensor, "batch_size num_nodes num_node_hidden_dims"],  # noqa: F722
        Float[torch.Tensor, "batch_size num_nodes num_nodes num_edge_hidden_dims"],  # noqa: F722
    ]:
        """Embed a set of graph inputs.

        Args:
            batch: A dictionary containing graph input features
                as well as `t`, a timestep sampled from [0, 1].
        Returns:
            Node embedding.
            Edge embedding.
        """
        t = batch["t"]
        node_indices = batch["node_indices"]
        fixed_node_mask = batch["fixed_node_mask"]
        self_conditioning_pos = batch["sc_pos_t"]
        self_conditioning_seq = batch.get("sc_aatype_t", None)
        asym_id = batch.get("asym_id", None)
        sym_id = batch.get("sym_id", None)
        entity_id = batch.get("entity_id", None)

        node_feats = []
        batch_size, num_nodes = node_indices.shape

        # set timestep e.g., to `epsilon=1e-5` for fixed residues
        t_embed = torch.tile(self.timestep_embedder(t)[:, None, :], (1, num_nodes, 1))
        t_embed[fixed_node_mask.bool()] = fixed_node_eps
        t_embed = torch.cat([t_embed, fixed_node_mask[..., None]], dim=-1)
        node_feats = [t_embed]
        edge_feats = [self.cross_concat(t_embed)]

        # compute relative position encoding
        relpos = self.relpos_embedder(
            residue_index=node_indices,
            asym_id=asym_id,
            sym_id=sym_id,
            entity_id=entity_id,
        )
        edge_feats.append(relpos.reshape([batch_size, num_nodes**2, -1]))

        # add existing per-node molecule type embeddings, if given
        if self.embedding_cfg.embed_molecule_type_conditioning:
            node_feats.append(self.molecule_type_embedder(batch["molecule_type_encoding"]))

        # add one-hot vector of the diffused input sequence, if given
        if self.diffuse_sequence:
            node_types = batch["node_types"].clone()
            # ensure nucleic acid types are in `aatype9` format
            node_types[batch["is_na_residue_mask"]] = convert_na_aatype6_to_aatype9(
                aatype=node_types[batch["is_na_residue_mask"]],
                deoxy_offset_mask=batch["node_deoxy"][batch["is_na_residue_mask"]],
            )
            node_feats.append(self.node_type_embedder(node_types))

        # curate self-conditioning distogram
        if self.embedding_cfg.embed_self_conditioning:
            sc_dgram = du.calc_distogram(
                self_conditioning_pos,
                self.embedding_cfg.min_bin,
                self.embedding_cfg.max_bin,
                self.embedding_cfg.num_bins,
            )
            edge_feats.append(sc_dgram.reshape([batch_size, num_nodes**2, -1]))
            # curate self-conditioning sequence embedding
            if self.diffuse_sequence:
                node_feats.append(self_conditioning_seq)

        # add one-hot vector embeddings of the diffused input sequence, if given
        if self.diffuse_sequence:
            row_onehot_node_types = (
                batch["onehot_node_types"] @ self.row_onehot_node_type_embedder.weight
            )[..., None, :]
            col_onehot_node_types = (
                batch["onehot_node_types"] @ self.col_onehot_node_type_embedder.weight
            )[..., None, :, :]
            onehot_node_types = row_onehot_node_types + col_onehot_node_types
            edge_feats.append(onehot_node_types.reshape([batch_size, num_nodes**2, -1]))

        # combine embeddings
        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())
        edge_embed = self.edge_embedder(torch.cat(edge_feats, dim=-1).float())
        edge_embed = edge_embed.reshape([batch_size, num_nodes, num_nodes, -1])

        return node_embed, edge_embed


class SE3ScoreNetwork(nn.Module):
    def __init__(self, model_cfg: DictConfig, ddpm: Any, sequence_ddpm: Optional[Any] = None):
        super().__init__()
        self.model_cfg = model_cfg

        self.score_embedding = ScoreEmbedding(model_cfg, ddpm.diffusion_cfg)
        self.score_model = FrameDiff(model_cfg, ddpm, sequence_ddpm=sequence_ddpm)
        self.sequence_ddpm = sequence_ddpm

    @jaxtyped
    @beartype
    def _apply_mask(
        self,
        value_diff: Float[torch.Tensor, "... num_value_dims"],  # noqa: F722
        value: Float[torch.Tensor, "... num_value_dims"],  # noqa: F722
        diff_mask: Float[torch.Tensor, "... 1"],  # noqa: F722
    ) -> Float[torch.Tensor, "batch_size num_nodes num_value_dims"]:  # noqa: F722
        return diff_mask * value_diff + (1 - diff_mask) * value

    @jaxtyped
    @beartype
    def forward(self, batch: BATCH_TYPE, plot_pred_positions: bool = False) -> Dict[str, Any]:
        """Forward computes the reverse diffusion conditionals `p(X ^ t | X ^ {t + 1})` for each
        item in the batch.

        Args:
            batch: A dictionary containing noised samples from the noising process,
                where the `T` timesteps are from `t = 1, ..., T`
                (i.e., not including the un-noised `X^0`)
            plot_pred_positions: Whether to plot the network's prediction atom37
                positions for debugging purposes.
        Returns:
            Dictionary of model outputs
        """
        # derive masks for nodes and edges, respectively
        node_mask = batch["node_mask"]
        edge_mask = batch["edge_mask"]
        fixed_node_mask = batch["fixed_node_mask"]
        is_protein_residue_mask = batch["is_protein_residue_mask"]
        is_na_residue_mask = batch["is_na_residue_mask"]
        protein_inputs_present = is_protein_residue_mask.any().item()
        na_inputs_present = is_na_residue_mask.any().item()

        # create initial embeddings of positional, relative, and chain indices as well as node types and representations
        init_node_embed, init_edge_embed = self.score_embedding(batch)

        batch["node_embed"] = init_node_embed * node_mask[..., None]
        batch["edge_embed"] = init_edge_embed * edge_mask[..., None]

        # execute primary network
        model_outputs = self.score_model(batch)

        # use the model's torsion angle predictions for side-chain reconstruction
        pred_torsions = model_outputs["torsions"]
        # protein residues #
        if protein_inputs_present:
            # mask torsion angles for protein residues
            pred_protein_torsions = pred_torsions[is_protein_residue_mask].view(
                pred_torsions.shape[0], -1, pred_torsions.shape[-1]
            )[..., : NUM_PROT_TORSIONS * 2]
            gt_protein_torsions = batch["torsion_angles_sin_cos"][is_protein_residue_mask].view(
                batch["torsion_angles_sin_cos"].shape[0],
                -1,
                *batch["torsion_angles_sin_cos"].shape[2:],
            )[..., 2, :]
            protein_diff_mask = 1 - fixed_node_mask[is_protein_residue_mask].view(
                fixed_node_mask.shape[0], -1, 1
            )
            pred_torsions[..., : NUM_PROT_TORSIONS * 2][
                is_protein_residue_mask
            ] = self._apply_mask(
                pred_protein_torsions, gt_protein_torsions, protein_diff_mask
            ).view(
                -1, NUM_PROT_TORSIONS * 2
            )
        # NA residues #
        if na_inputs_present:
            # mask torsion angles for nucleic acid residues
            pred_na_torsions = pred_torsions[is_na_residue_mask].view(
                pred_torsions.shape[0], -1, pred_torsions.shape[-1]
            )[..., : NUM_NA_TORSIONS * 2]
            gt_na_torsions = batch["torsion_angles_sin_cos"][is_na_residue_mask].view(
                batch["torsion_angles_sin_cos"].shape[0],
                -1,
                *batch["torsion_angles_sin_cos"].shape[2:],
            )[..., :NUM_NA_TORSIONS, :]
            na_diff_mask = 1 - fixed_node_mask[is_na_residue_mask].view(
                fixed_node_mask.shape[0], -1, 1
            )
            pred_torsions[is_na_residue_mask] = self._apply_mask(
                pred_na_torsions, gt_na_torsions.flatten(-2, -1), na_diff_mask
            ).view(-1, pred_torsions.shape[-1])

        # derive frames as Tensors of shape [batch_size, num_nodes, 7]
        pred_outputs = {
            "torsions": pred_torsions,
            "rot_score": model_outputs["rot_score"],
            "trans_score": model_outputs["trans_score"],
            "pred_node_types": model_outputs.get("pred_node_types", None),
        }
        rigids_pred = model_outputs["final_rigids"]
        pred_outputs["rigids"] = rigids_pred.to_tensor_7()
        bb_representations = all_atom.compute_backbone(
            bb_rigids=rigids_pred,
            torsions=pred_torsions,
            is_protein_residue_mask=is_protein_residue_mask,
            is_na_residue_mask=is_na_residue_mask,
        )
        pred_outputs["atom37"] = bb_representations[0].to(rigids_pred.device)
        pred_outputs["atom23"] = bb_representations[-1].to(rigids_pred.device)
        pred_outputs["atom37_supervised_mask"] = bb_representations[2].to(rigids_pred.device)

        if plot_pred_positions:
            with torch.no_grad():
                # optionally, debug the initial frames to make sure they are constructed correctly
                # rigids_pred = ru.Rigid.from_tensor_7(torch.clone(batch["rigids_t"].type(torch.float32)))
                # bb_representations = all_atom.compute_backbone(
                #     bb_rigids=rigids_pred,
                #     torsions=pred_torsions,
                #     is_protein_residue_mask=is_protein_residue_mask,
                #     is_na_residue_mask=is_na_residue_mask,
                # )
                # pred_outputs["atom37"] = bb_representations[0].to(rigids_pred.device)
                # pred_outputs["atom23"] = bb_representations[-1].to(rigids_pred.device)
                # pred_outputs["atom37_supervised_mask"] = bb_representations[2].to(rigids_pred.device)

                # collect unpadded outputs for the first batch element
                unpad_is_protein_residue_mask = is_protein_residue_mask[0].tolist()
                unpad_is_na_residue_mask = is_na_residue_mask[0].tolist()
                unpad_pred_pos = pred_outputs["atom37"][0].cpu().detach().numpy()
                unpad_complex_restype = batch["aatype"][0].cpu().numpy()
                # unpad_complex_restype[unpad_is_na_residue_mask] += vocabulary.protein_restype_num + 1
                # unpad_complex_restype[
                #     unpad_is_protein_residue_mask
                # ] = 0  # denote the default amino acid type (i.e., Alanine)
                # unpad_complex_restype[
                #     unpad_is_na_residue_mask
                # ] = 21  # denote the default nucleic acid type (i.e., Adenine)
                unpad_node_chain_indices = batch["node_chain_indices"][0].cpu().numpy()
                unpad_fixed_mask = batch["fixed_node_mask"][0].cpu().numpy()
                b_factors = np.tile(1 - unpad_fixed_mask[..., None], 37) * 100

                # construct output metadata
                current_date = datetime.date.today().strftime("%Y-%m-%d")
                base_temp_dir = os.path.join(
                    tempfile.gettempdir(), f"training_samples_{current_date}"
                )
                os.makedirs(base_temp_dir, exist_ok=True)
                sample_number = 0
                file_extension = "pdb"
                existing_files = os.listdir(base_temp_dir)
                sample_numbers = []
                pattern = re.compile(r"^training_sample_(\d+)\.pdb$")
                for filename in existing_files:
                    match = pattern.match(filename)
                    if match:
                        sample_numbers.append(int(match.group(1)))
                if sample_numbers:
                    sample_number = max(sample_numbers) + 1
                unique_output_filename = f"training_sample_{sample_number}.{file_extension}"
                unique_output_filepath = os.path.join(base_temp_dir, unique_output_filename)

                # save outputs as PDB files
                (
                    saved_protein_pdb_path,
                    saved_na_pdb_path,
                    saved_protein_na_pdb_path,
                ) = au.write_complex_to_pdbs(
                    complex_pos=unpad_pred_pos,
                    output_filepath=unique_output_filepath,
                    restype=unpad_complex_restype,
                    chain_index=unpad_node_chain_indices,
                    b_factors=b_factors,
                    is_protein_residue_mask=unpad_is_protein_residue_mask,
                    is_na_residue_mask=unpad_is_na_residue_mask,
                    no_indexing=True,
                )
                log.info(
                    f"Since `plot_pred_positions` is `True`, logged `saved_protein_na_pdb_path` to {saved_protein_na_pdb_path}"
                )

        return pred_outputs
