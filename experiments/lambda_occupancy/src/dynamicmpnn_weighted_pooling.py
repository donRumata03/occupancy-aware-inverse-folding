from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


_ACTIVE_CONFORMATION_WEIGHTS: ContextVar[Any] = ContextVar("dynamicmpnn_conformation_weights", default=None)
_PATCHED = False


def patch_dynamicmpnn_weighted_pooling() -> None:
    """Install inference-time weighted pooling hooks into DynamicMPNN.

    The upstream DynamicMPNN implementation performs symmetric masked-mean pooling over
    conformations. This patch keeps that default behavior when no weights are provided and
    switches to weighted masked means when `eval.conformation_weights` is set.
    """

    global _PATCHED
    if _PATCHED:
        return

    import torch
    import dynamicmpnn.eval.sampling as sampling
    import dynamicmpnn.modules.models.models as models

    original_pool_conformations = models.pool_conformations
    original_pool_edges_conformations = models.pool_edges_conformations
    original_sample = models.DynamicMPNN.sample
    original_forward = models.DynamicMPNN.forward
    original_pool_for_sampling = models.DynamicMPNN.pool_for_sampling
    original_pool_for_training = models.DynamicMPNN.pool_for_training

    def active_weights(explicit_weights: Any = None) -> Any:
        return explicit_weights if explicit_weights is not None else _ACTIVE_CONFORMATION_WEIGHTS.get()

    def weight_tensor(conformation_weights: Any, n_conformations: int, reference: torch.Tensor) -> torch.Tensor | None:
        if conformation_weights is None:
            return None
        weights = torch.as_tensor(conformation_weights, device=reference.device, dtype=reference.dtype)
        if weights.ndim == 2 and weights.shape[0] == 1:
            weights = weights.squeeze(0)
        if weights.ndim != 1 or weights.numel() != n_conformations:
            raise ValueError(
                f"conformation_weights must have shape [{n_conformations}] for this inference path; "
                f"got {tuple(weights.shape)}"
            )
        return weights

    def weighted_pool_conformations(node_features, virtual_mask, conformation_weights=None):
        weights = weight_tensor(active_weights(conformation_weights), node_features[0].shape[1], node_features[0])
        if weights is None:
            return original_pool_conformations(node_features, virtual_mask)

        s, v = node_features
        mask_s = virtual_mask.unsqueeze(-1).to(dtype=s.dtype)
        mask_v = virtual_mask.unsqueeze(-1).unsqueeze(-1).to(dtype=v.dtype)
        weight_s = weights.view(1, -1, 1)
        weight_v = weights.view(1, -1, 1, 1)

        denom_s = (mask_s * weight_s).sum(dim=1).clamp(min=1e-12)
        denom_v = (mask_v * weight_v).sum(dim=1).clamp(min=1e-12)
        s_pooled = (s * mask_s * weight_s).sum(dim=1) / denom_s
        v_pooled = (v * mask_v * weight_v).sum(dim=1) / denom_v
        return s_pooled, v_pooled

    def weighted_pool_edges_conformations(edge_features, edge_mask, conformation_weights=None):
        weights = weight_tensor(active_weights(conformation_weights), edge_features[0].shape[1], edge_features[0])
        if weights is None:
            return original_pool_edges_conformations(edge_features, edge_mask)

        s, v = edge_features
        mask_s = edge_mask.unsqueeze(-1).to(dtype=s.dtype)
        mask_v = mask_s.unsqueeze(-1)
        weight_s = weights.view(1, -1, 1)
        weight_v = weights.view(1, -1, 1, 1)

        denom_s = (mask_s * weight_s).sum(dim=1).clamp(min=1e-12)
        denom_v = denom_s.unsqueeze(-1)
        s_pooled = (s * mask_s * weight_s).sum(dim=1) / denom_s
        v_pooled = (v * mask_v * weight_v).sum(dim=1) / denom_v
        return s_pooled, v_pooled

    def weighted_stack_pool(features, mask, conformation_weights):
        weights = weight_tensor(conformation_weights, features[0].shape[0], features[0])
        if weights is None:
            return None
        s, v = features
        mask_s = mask.unsqueeze(-1).to(dtype=s.dtype)
        mask_v = mask.unsqueeze(-1).unsqueeze(-1).to(dtype=v.dtype)
        weight_s = weights.view(-1, 1, 1)
        weight_v = weights.view(-1, 1, 1, 1)
        denom_s = (mask_s * weight_s).sum(dim=0).clamp(min=1e-12)
        denom_v = (mask_v * weight_v).sum(dim=0).clamp(min=1e-12)
        return (s * mask_s * weight_s).sum(dim=0) / denom_s, (v * mask_v * weight_v).sum(dim=0) / denom_v

    @contextmanager
    def weights_context(conformation_weights):
        token = _ACTIVE_CONFORMATION_WEIGHTS.set(conformation_weights)
        try:
            yield
        finally:
            _ACTIVE_CONFORMATION_WEIGHTS.reset(token)

    def patched_sample(self, batch, logit_bias=None, return_logits=False, conformation_weights=None):
        with weights_context(conformation_weights):
            return original_sample(self, batch, logit_bias=logit_bias, return_logits=return_logits)

    def patched_forward(self, batch, conformation_weights=None):
        with weights_context(conformation_weights):
            return original_forward(self, batch)

    def patched_pool_for_sampling(
        self,
        h_V_,
        h_E_,
        edge_index_local,
        homomer_index,
        virtual_mask,
        decoder_edge_mask,
        n_edges,
        original_indices,
    ):
        weights = active_weights()
        if weights is None:
            return original_pool_for_sampling(
                self,
                h_V_,
                h_E_,
                edge_index_local,
                homomer_index,
                virtual_mask,
                decoder_edge_mask,
                n_edges,
                original_indices,
            )

        import torch
        import dynamicmpnn.modules.models.models as models

        n_chains = len(h_E_[0]) // n_edges
        chunk_size = n_edges
        h_E_chunks = [
            (h_E_[0][i * chunk_size : (i + 1) * chunk_size], h_E_[1][i * chunk_size : (i + 1) * chunk_size])
            for i in range(n_chains)
        ]
        edge_mask_chunks = [decoder_edge_mask[i * chunk_size : (i + 1) * chunk_size] for i in range(n_chains)]

        homo_ids = torch.unique(homomer_index)[torch.unique(homomer_index) != models.HOMOMER_NEGATIVE]
        chain_indices = [torch.where(homomer_index == hid)[0] for hid in homo_ids]
        all_extractions = [
            (h_V_[0][chain_idx_list], h_V_[1][chain_idx_list], virtual_mask[chain_idx_list], virtual_mask[chain_idx_list])
            for chain_idx_list in chain_indices
        ]

        h_E_stacked = (
            torch.stack([chunk[0] for chunk in h_E_chunks], dim=0),
            torch.stack([chunk[1] for chunk in h_E_chunks], dim=0),
        )
        edge_mask_stacked = torch.stack(edge_mask_chunks, dim=0)
        h_E_pooled = weighted_stack_pool(h_E_stacked, edge_mask_stacked, weights)

        h_V_s_chains, h_V_v_chains, mask_s_chains, _ = zip(*all_extractions)
        h_V_stacked = (torch.stack(h_V_s_chains, dim=0), torch.stack(h_V_v_chains, dim=0))
        mask_stacked = torch.stack(mask_s_chains, dim=0)
        h_V_pooled = weighted_stack_pool(h_V_stacked, mask_stacked, weights)

        return h_V_pooled, h_E_pooled, edge_index_local, original_indices

    def patched_pool_for_training(
        self,
        h_V_,
        h_E_,
        homomer_index,
        virtual_mask,
        decoder_edge_mask,
        n_edges,
    ):
        weights = active_weights()
        if weights is None:
            return original_pool_for_training(
                self,
                h_V_,
                h_E_,
                homomer_index,
                virtual_mask,
                decoder_edge_mask,
                n_edges,
            )

        h_V_pooled, h_E_pooled, _, _ = patched_pool_for_sampling(
            self,
            h_V_,
            h_E_,
            edge_index_local=None,
            homomer_index=homomer_index,
            virtual_mask=virtual_mask,
            decoder_edge_mask=decoder_edge_mask,
            n_edges=n_edges,
            original_indices=None,
        )
        return h_V_pooled, h_E_pooled, None

    def patched_load_model_and_sample_batch(checkpoint_path, cfg, batch, num_samples, device="auto"):
        torch_device = sampling.resolve_device(device)
        model = sampling._load_model_for_sampling(checkpoint_path, cfg, num_samples)
        model = model.to(torch_device)

        weights = None
        eval_cfg = cfg.get("eval")
        if eval_cfg is not None:
            weights = eval_cfg.get("conformation_weights")

        batch = batch.to(torch_device)
        with torch.inference_mode():
            sample_tensor, _, _, _ = model.sample(batch, return_logits=True, conformation_weights=weights)
        return sampling.decode_samples(sample_tensor)

    models.pool_conformations = weighted_pool_conformations
    models.pool_edges_conformations = weighted_pool_edges_conformations
    models.DynamicMPNN.sample = patched_sample
    models.DynamicMPNN.forward = patched_forward
    models.DynamicMPNN.pool_for_sampling = patched_pool_for_sampling
    models.DynamicMPNN.pool_for_training = patched_pool_for_training
    sampling.load_model_and_sample_batch = patched_load_model_and_sample_batch
    _PATCHED = True
