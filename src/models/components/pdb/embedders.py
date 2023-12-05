# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for MMDiff (https://github.com/Profluent-Internships/MMDiff):
# -------------------------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from beartype import beartype
from beartype.typing import Optional

from src.data.components.pdb.data_transforms import make_one_hot
from src.models.components.pdb.framediff import Linear


class RelPosEncoder(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        max_relative_idx: int,
        max_relative_chain: int = 0,
        use_chain_relative: bool = False,
    ):
        super().__init__()
        self.max_relative_idx = max_relative_idx
        self.max_relative_chain = max_relative_chain
        self.use_chain_relative = use_chain_relative
        self.num_bins = 2 * max_relative_idx + 2
        if max_relative_chain > 0:
            self.num_bins += 2 * max_relative_chain + 2
        if use_chain_relative:
            self.num_bins += 1

        self.linear_relpos = Linear(self.num_bins, embedding_size)

    @beartype
    def forward(
        self,
        residue_index: torch.Tensor,
        asym_id: Optional[torch.Tensor] = None,
        sym_id: Optional[torch.Tensor] = None,
        entity_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        d = residue_index[..., None] - residue_index[..., None, :]

        if asym_id is None:
            # compute relative position encoding according to AlphaFold's `relpos` algorithm
            boundaries = torch.arange(
                start=-self.max_relative_idx, end=self.max_relative_idx + 1, device=d.device
            )
            reshaped_bins = boundaries.view(((1,) * len(d.shape)) + (len(boundaries),))
            d = d[..., None] - reshaped_bins
            d = torch.abs(d)
            d = torch.argmin(d, dim=-1)
            d = nn.functional.one_hot(d, num_classes=len(boundaries)).float()
            d = d.to(residue_index.dtype)
            rel_feat = d
        else:
            # compute relative position encoding according to AlphaFold-Multimer's `relpos` algorithm
            rel_feats = []
            asym_id_same = torch.eq(asym_id[..., None], asym_id[..., None, :])
            offset = residue_index[..., None] - residue_index[..., None, :]

            clipped_offset = torch.clamp(
                input=offset + self.max_relative_idx, min=0, max=(2 * self.max_relative_idx)
            )

            final_offset = torch.where(
                condition=asym_id_same,
                input=clipped_offset,
                other=((2 * self.max_relative_idx + 1) * torch.ones_like(clipped_offset)),
            )

            rel_pos = make_one_hot(x=final_offset, num_classes=(2 * self.max_relative_idx + 2))
            rel_feats.append(rel_pos)

            if self.use_chain_relative:
                entity_id_same = torch.eq(entity_id[..., None], entity_id[..., None, :])
                rel_feats.append(entity_id_same.type(rel_pos.dtype)[..., None])

            if self.max_relative_chain > 0:
                rel_sym_id = sym_id[..., None] - sym_id[..., None, :]
                max_rel_chain = self.max_relative_chain

                clipped_rel_chain = torch.clamp(
                    input=rel_sym_id + max_rel_chain, min=0, max=2 * max_rel_chain
                )

                if not self.use_chain_relative:
                    # ensure `entity_id_same` is constructed for `rel_chain`
                    entity_id_same = torch.eq(entity_id[..., None], entity_id[..., None, :])

                final_rel_chain = torch.where(
                    condition=entity_id_same,
                    input=clipped_rel_chain,
                    other=(2 * max_rel_chain + 1) * torch.ones_like(clipped_rel_chain),
                )
                rel_chain = make_one_hot(
                    x=final_rel_chain, num_classes=2 * self.max_relative_chain + 2
                )
                rel_feats.append(rel_chain)

            rel_feat = torch.cat(rel_feats, dim=-1)

        return self.linear_relpos(rel_feat)
