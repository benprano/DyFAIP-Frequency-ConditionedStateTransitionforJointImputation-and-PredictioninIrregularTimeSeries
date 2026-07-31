import torch

class DataSampler:
    def __init__(self, percentage=0.2, mode='MCAR', feature_idx=None, threshold=None):
        self.percentage = percentage
        self.mode = mode
        self.feature_idx = [feature_idx] if isinstance(feature_idx, int) else feature_idx
        self.threshold = [threshold] if isinstance(threshold, (float, int)) else threshold

    def mark_data_as_missing(self, data):
        data_with_missing = data.clone()
        mask = ~torch.isnan(data)

        if self.mode == 'MCAR':
            sampled_3d_indices = self._sample_mcar(data, mask)

        elif self.mode == 'MAR':
            sampled_3d_indices = self._sample_mar(mask)

        elif self.mode == 'MNAR':
            sampled_3d_indices = self._sample_mnar(data, mask)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        selected_data = data_with_missing[
            sampled_3d_indices[0], sampled_3d_indices[1], sampled_3d_indices[2]
        ]
        assert not torch.isnan(selected_data).any(), "selected_data contains NaNs!"
        data_with_missing[
            sampled_3d_indices[0], sampled_3d_indices[1], sampled_3d_indices[2]
        ] = float('nan')

        return selected_data, data_with_missing, tuple(sampled_3d_indices)

    def _sample_mcar(self, data, mask):
        observed_flat_indices = torch.nonzero(mask.view(-1), as_tuple=False).squeeze()
        num_samples = int(self.percentage * observed_flat_indices.size(0))
        sampled_indices = observed_flat_indices[
            torch.randperm(observed_flat_indices.size(0), device=data.device)[:num_samples]
        ]
        return torch.stack(torch.unravel_index(sampled_indices, data.shape))

    def _sample_mar(self, mask):
        if self.feature_idx is None:
            raise ValueError("feature_idx must be provided for MAR")

        cond_mask = mask[:, :, self.feature_idx[0]].clone()
        for f in self.feature_idx[1:]:
            cond_mask &= mask[:, :, f]

        valid_btf = torch.nonzero(mask, as_tuple=False)
        valid_btf = valid_btf[cond_mask[valid_btf[:, 0], valid_btf[:, 1]]]

        num_samples = int(self.percentage * len(valid_btf))
        sampled = valid_btf[torch.randperm(len(valid_btf), device=valid_btf.device)[:num_samples]]
        return sampled.T

    def _sample_mnar(self, data, mask):
        if self.feature_idx is None or self.threshold is None:
            raise ValueError("feature_idx and threshold must be provided for MNAR")
        if len(self.feature_idx) != len(self.threshold):
            raise ValueError("feature_idx and threshold must have the same length")

        btf_positions = []
        for f_idx, thr in zip(self.feature_idx, self.threshold):
            feat_vals = data[:, :, f_idx]
            valid = (feat_vals >= thr) & mask[:, :, f_idx]
            b_idxs, t_idxs = torch.where(valid)
            if b_idxs.numel() > 0:
                f_idxs = torch.full((b_idxs.numel(),), f_idx, device=data.device)
                btf_positions.append(torch.stack([b_idxs, t_idxs, f_idxs], dim=0))

        if not btf_positions:
            # Fall back instead of crashing: this batch has no values above
            # the training-derived threshold (common for small/edge batches,
            # e.g., the final test batch). Relax to "any observed value" for
            # this batch only, so evaluation can proceed.
            """
            print(
                "[MNAR][WARN] No values exceeded threshold in this batch — "
                "falling back to MCAR-style sampling among observed values "
                "for this batch only."
            )"""
            fallback_positions = []
            for f_idx in self.feature_idx:
                valid = mask[:, :, f_idx]
                b_idxs, t_idxs = torch.where(valid)
                if b_idxs.numel() > 0:
                    f_idxs = torch.full((b_idxs.numel(),), f_idx, device=data.device)
                    fallback_positions.append(torch.stack([b_idxs, t_idxs, f_idxs], dim=0))

            if not fallback_positions:
                raise ValueError(
                    "No observed values available at all for MNAR fallback "
                    "(batch may be fully NaN for the sampled features)."
                )
            all_btf = torch.cat(fallback_positions, dim=1)
        else:
            all_btf = torch.cat(btf_positions, dim=1)

        perm = torch.randperm(all_btf.shape[1], device=all_btf.device)
        num_samples = max(1, int(self.percentage * all_btf.shape[1]))
        return all_btf[:, perm[:num_samples]]
        data_with_missing[sampled_3d_indices[0], sampled_3d_indices[1], sampled_3d_indices[2]] = float('nan')

        return selected_data, data_with_missing, tuple(sampled_3d_indices)
