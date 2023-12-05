# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from se3_diffusion (https://github.com/jasonkyuyim/se3_diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------
"""Fork of Openfold's IPA."""

import math

import numpy as np
import torch
import torch.nn as nn
from beartype.typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from omegaconf import DictConfig
from scipy.stats import truncnorm

from src.data.components.pdb import all_atom
from src.data.components.pdb.pdb_na_dataset import (
    BATCH_TYPE,
    NUM_PROTEIN_NA_ONEHOT_AATYPE_CLASSES,
)
from src.data.components.pdb.rigid_utils import Rigid


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def compute_angles(ca_pos, pts):
    batch_size, num_res, num_heads, num_pts, _ = pts.shape
    calpha_vecs = (ca_pos[:, :, None, :] - ca_pos[:, None, :, :]) + 1e-10
    calpha_vecs = torch.tile(calpha_vecs[:, :, :, None, None, :], (1, 1, 1, num_heads, num_pts, 1))
    ipa_pts = pts[:, :, None, :, :, :] - torch.tile(
        ca_pos[:, :, None, None, None, :], (1, 1, num_res, num_heads, num_pts, 1)
    )
    phi_angles = all_atom.calculate_neighbor_angles(
        calpha_vecs.reshape(-1, 3), ipa_pts.reshape(-1, 3)
    ).reshape(batch_size, num_res, num_res, num_heads, num_pts)
    return phi_angles


class Linear(nn.Linear):
    """A Linear layer with built-in nonstandard initializations. Called just like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super().__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                lecun_normal_init_(self.weight)
            elif init == "relu":
                he_normal_init_(self.weight)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")


class StructureModuleTransition(nn.Module):
    def __init__(self, c):
        super().__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(self.c)

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        s = s + s_initial
        s = self.ln(s)

        return s


class EdgeTransition(nn.Module):
    def __init__(
        self, *, node_embed_size, edge_embed_in, edge_embed_out, num_layers=2, node_dilation=2
    ):
        super().__init__()

        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = Linear(node_embed_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, node_embed, edge_embed):
        node_embed = self.initial_embed(node_embed)
        batch_size, num_res, _ = node_embed.shape
        edge_bias = torch.cat(
            [
                torch.tile(node_embed[:, :, None, :], (1, 1, num_res, 1)),
                torch.tile(node_embed[:, None, :, :], (1, num_res, 1, 1)),
            ],
            axis=-1,
        )
        edge_embed = torch.cat([edge_embed, edge_bias], axis=-1).reshape(
            batch_size * num_res**2, -1
        )
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        edge_embed = edge_embed.reshape(batch_size, num_res, num_res, -1)
        return edge_embed


class InvariantPointAttention(nn.Module):
    """Implements Algorithm 22."""

    def __init__(
        self,
        ipa_conf,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            num_heads:
                Number of attention heads
            num_qk_points:
                Number of query/key points to generate
            num_v_points:
                Number of value points to generate
        """
        super().__init__()
        self._ipa_conf = ipa_conf

        self.c_s = ipa_conf.c_s
        self.c_z = ipa_conf.c_z
        self.c_hidden = ipa_conf.c_hidden
        self.num_heads = ipa_conf.num_heads
        self.num_qk_points = ipa_conf.num_qk_points
        self.num_v_points = ipa_conf.num_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.num_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.num_heads * self.num_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.num_heads * (self.num_qk_points + self.num_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        self.linear_b = Linear(self.c_z, self.num_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        self.head_weights = nn.Parameter(torch.zeros(ipa_conf.num_heads))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.c_z // 4 + self.c_hidden + self.num_v_points * 4
        self.linear_out = Linear(self.num_heads * concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        r: Rigid,
        mask: torch.Tensor,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.num_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.num_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(q_pts.shape[:-2] + (self.num_heads, self.num_qk_points, 3))

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.num_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(kv_pts, [self.num_qk_points, self.num_v_points], dim=-2)

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])

        if _offload_inference:
            z[0] = z[0].cpu()

        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1))

        # [*, N_res, N_res, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_displacement**2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(1.0 / (3 * (self.num_qk_points * 9.0 / 2)))
        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(a, v.transpose(-2, -3).to(dtype=a.dtype)).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v]
        o_pt = torch.sum(
            (a[..., None, :, :, None] * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(o_pt_dists, 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if _offload_inference:
            z[0] = z[0].to(o_pt.device)

        # [*, N_res, H, C_z // 4]
        pair_z = self.down_z(z[0]).to(dtype=a.dtype)
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)

        # [*, N_res, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]

        # [*, N_res, C_s]
        s = self.linear_out(torch.cat(o_feats, dim=-1).to(dtype=z[0].dtype))

        return s


class AngleResnetBlock(nn.Module):
    def __init__(self, c_hidden: int):
        """
        Args:
            c_hidden:
                Hidden channel dimension
        """
        super().__init__()

        self.c_hidden = c_hidden

        self.linear_1 = Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init="final")

        self.relu = nn.ReLU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        s_initial = a

        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)

        return a + s_initial


class AngleResnet(nn.Module):
    """Implements Algorithm 20, lines 11-14, in reference to AlphaFold 2's torsion angle
    predictions."""

    def __init__(
        self, c_in: int, c_hidden: int, num_blocks: int, num_angles: int, epsilon: float = 1e-8
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            num_blocks:
                Number of resnet blocks
            num_angles:
                Number of torsion angles to generate
            epsilon:
                Small constant for normalization
        """
        super().__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.num_blocks = num_blocks
        self.num_angles = num_angles
        self.eps = epsilon

        self.linear_in = Linear(self.c_in, self.c_hidden)
        self.linear_initial = Linear(self.c_in, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.num_blocks):
            layer = AngleResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.linear_out = Linear(self.c_hidden, self.num_angles * 2)

        self.relu = nn.ReLU()

    def forward(
        self,
        s: torch.Tensor,
        s_initial: torch.Tensor,
        is_first_molecule_type_residue_mask: Optional[torch.Tensor] = None,
        is_second_molecule_type_residue_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the
                StructureModule
            is_first_molecule_type_residue_mask:
                [*] optional mask denoting which residues are
                of the first molecule type (e.g., protein) and which are not
            is_second_molecule_type_residue_mask:
                [*] optional mask denoting which residues are
                of the second molecule type (e.g., nucleic acid) and which are not
        Returns:
            [*, num_angles, 2] predicted angles
        """
        # [*, C_hidden]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        for layer in self.layers:
            s = layer(s)

        s = self.relu(s)

        # [*, num_angles * 2]
        s = self.linear_out(s)

        # [*, num_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        if (
            is_first_molecule_type_residue_mask is not None
            or is_second_molecule_type_residue_mask is not None
        ):
            is_first_molecule_type_residue_mask_ = (
                is_first_molecule_type_residue_mask
                if is_first_molecule_type_residue_mask is not None
                else ~is_second_molecule_type_residue_mask
            )
            is_second_molecule_type_residue_mask_ = (
                is_second_molecule_type_residue_mask
                if is_second_molecule_type_residue_mask is not None
                else ~is_first_molecule_type_residue_mask
            )
            first_molecule_type_norm_denom = torch.sqrt(
                torch.clamp(
                    torch.sum(s[is_first_molecule_type_residue_mask_] ** 2, dim=-1, keepdim=True),
                    min=self.eps,
                )
            )
            second_molecule_type_norm_denom = torch.sqrt(
                torch.clamp(
                    torch.sum(s[is_second_molecule_type_residue_mask_] ** 2, dim=-1, keepdim=True),
                    min=self.eps,
                )
            )
            s[is_first_molecule_type_residue_mask_] = (
                s[is_first_molecule_type_residue_mask_] / first_molecule_type_norm_denom
            )
            s[is_second_molecule_type_residue_mask_] = (
                s[is_second_molecule_type_residue_mask_] / second_molecule_type_norm_denom
            )
        else:
            norm_denom = torch.sqrt(
                torch.clamp(
                    torch.sum(s**2, dim=-1, keepdim=True),
                    min=self.eps,
                )
            )
            s = s / norm_denom

        return unnormalized_s, s


class SequenceMLP(nn.Module):
    def __init__(self, c, num_residue_types):
        super().__init__()

        self.c = c
        self.num_residue_types = num_residue_types

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_final = Linear(self.c, self.num_residue_types, init="default")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)

        s = s + s_initial
        s_final = self.linear_final(s)

        return s_final


class ScoreLayer(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out):
        super().__init__()

        self.linear_1 = Linear(dim_in, dim_hid, init="relu")
        self.linear_2 = Linear(dim_hid, dim_hid)
        self.linear_3 = Linear(dim_hid, dim_out, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = s + s_initial
        s = self.linear_3(s)
        return s


class BackboneUpdate(nn.Module):
    """Implements part of Algorithm 23."""

    def __init__(self, c_s):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super().__init__()

        self.c_s = c_s

        self.linear = Linear(self.c_s, 6, init="final")

    def forward(self, s: torch.Tensor):
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector
        """
        # [*, 6]
        update = self.linear(s)

        return update


class FrameDiff(nn.Module):
    def __init__(self, model_conf: DictConfig, ddpm: Any, sequence_ddpm: Optional[Any] = None):
        super().__init__()
        self._model_conf = model_conf
        self._diffusion_conf = ddpm.diffusion_cfg
        ipa_conf = model_conf.ipa
        self._ipa_conf = ipa_conf
        self.ddpm = ddpm
        self.sequence_ddpm = sequence_ddpm
        self.diffuse_sequence = (
            hasattr(ddpm.diffusion_cfg, "diffuse_sequence") and ddpm.diffusion_cfg.diffuse_sequence
        )

        self.trunk = nn.ModuleDict()

        for b in range(self._model_conf.num_layers):
            self.trunk[f"ipa_{b}"] = InvariantPointAttention(ipa_conf)
            self.trunk[f"ipa_ln_{b}"] = nn.LayerNorm(ipa_conf.c_s)
            self.trunk[f"skip_embed_{b}"] = Linear(
                self._model_conf.node_hidden_dim, self._model_conf.c_skip, init="final"
            )
            tfmr_in = ipa_conf.c_s + self._model_conf.c_skip
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._model_conf.tfmr.num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False,
            )
            self.trunk[f"seq_tfmr_{b}"] = torch.nn.TransformerEncoder(
                # Note: Currently with PyTorch 2.0, `enable_nested_tensor` must be `False` to
                # avoid an inference-time token removal bug that occurs with `src_key_padding_mask`
                tfmr_layer,
                self._model_conf.tfmr.num_layers,
                enable_nested_tensor=False,
            )
            self.trunk[f"post_tfmr_{b}"] = Linear(tfmr_in, ipa_conf.c_s, init="final")
            self.trunk[f"node_transition_{b}"] = StructureModuleTransition(c=ipa_conf.c_s)
            self.trunk[f"bb_update_{b}"] = BackboneUpdate(ipa_conf.c_s)

            if b < model_conf.num_layers - 1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_hidden_dim
                self.trunk[f"edge_transition_{b}"] = EdgeTransition(
                    node_embed_size=ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_hidden_dim,
                )

        self.torsion_pred = AngleResnet(
            ipa_conf.c_s, ipa_conf.c_resnet, ipa_conf.num_resnet_blocks, model_conf.num_angles
        )

        if self.diffuse_sequence:
            self.sequence_pred = SequenceMLP(ipa_conf.c_s, NUM_PROTEIN_NA_ONEHOT_AATYPE_CLASSES)

    def scale_rigids(self, x: Any) -> Any:
        return x.apply_trans_fn(lambda x_: x_ * self._model_conf.ipa.coordinate_scaling)

    def unscale_rigids(self, x: Any) -> Any:
        return x.apply_trans_fn(lambda x_: x_ / self._model_conf.ipa.coordinate_scaling)

    def forward(self, batch: BATCH_TYPE) -> Dict[str, Any]:
        node_embed = batch["node_embed"]
        edge_embed = batch["edge_embed"]

        node_mask = batch["node_mask"].type(torch.float32)
        diffuse_mask = (1 - batch["fixed_node_mask"].type(torch.float32)) * node_mask
        edge_mask = node_mask[..., None] * node_mask[..., None, :]
        init_frames = batch["rigids_t"].type(torch.float32)

        curr_rigids = Rigid.from_tensor_7(torch.clone(init_frames))
        init_rigids = Rigid.from_tensor_7(init_frames)

        # Main trunk
        curr_rigids = self.scale_rigids(curr_rigids)
        init_node_embed = node_embed * node_mask[..., None]
        node_embed = node_embed * node_mask[..., None]
        for b in range(self._model_conf.num_layers):
            ipa_embed = self.trunk[f"ipa_{b}"](node_embed, edge_embed, curr_rigids, node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f"ipa_ln_{b}"](node_embed + ipa_embed)
            seq_tfmr_in = torch.cat(
                [node_embed, self.trunk[f"skip_embed_{b}"](init_node_embed)], dim=-1
            )
            seq_tfmr_out = self.trunk[f"seq_tfmr_{b}"](
                seq_tfmr_in, src_key_padding_mask=((1 - node_mask).bool())
            )
            node_embed = node_embed + self.trunk[f"post_tfmr_{b}"](seq_tfmr_out)
            node_embed = self.trunk[f"node_transition_{b}"](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f"bb_update_{b}"](node_embed * diffuse_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(rigid_update, diffuse_mask[..., None])

            if b < self._model_conf.num_layers - 1:
                edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]
        rot_score = self.ddpm.calc_rot_score(
            init_rigids.get_rots(), curr_rigids.get_rots(), batch["t"]
        )
        rot_score = rot_score * node_mask[..., None]

        curr_rigids = self.unscale_rigids(curr_rigids)
        trans_score = self.ddpm.calc_trans_score(
            init_rigids.get_trans(),
            curr_rigids.get_trans(),
            batch["t"][:, None, None],
            use_torch=True,
        )
        trans_score = trans_score * node_mask[..., None]
        _, pred_torsions = self.torsion_pred(
            node_embed,
            init_node_embed,
            is_first_molecule_type_residue_mask=batch["is_protein_residue_mask"],
            is_second_molecule_type_residue_mask=batch["is_na_residue_mask"],
        )
        model_out = {
            "torsions": pred_torsions.flatten(start_dim=-2),
            "rot_score": rot_score,
            "trans_score": trans_score,
            "final_rigids": curr_rigids,
        }
        if self.diffuse_sequence:
            model_out["pred_node_types"] = self.sequence_pred(node_embed) * node_mask[..., None]
        return model_out
