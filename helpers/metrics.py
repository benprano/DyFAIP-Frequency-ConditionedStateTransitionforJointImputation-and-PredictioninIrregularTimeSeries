
from typing import Any, Dict, List, Optional, Tuple
import os
import torch
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score, f1_score, mean_absolute_error,
    mean_squared_error, r2_score, roc_curve, auc
)
from tqdm import tqdm


class TrainerMetrics:
    """Modular metrics computation for training pipeline."""

    def __init__(self, input_dim: int):
        self.input_dim = input_dim

    @staticmethod
    def adjusted_r2(actual: np.ndarray, predicted: np.ndarray,
                    n_samples: np.int64, n_features: np.int64) -> float:
        """Compute adjusted R² score."""
        r2 = r2_score(actual, predicted)
        return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features)

    def compute_regression_metrics(self, targets: List[np.ndarray],
                                   predicted: List[np.ndarray],
                                   rescale_params: Dict[str, float]) -> List[List[float]]:
        """Compute regression metrics (RMSE, MAE, R², Adjusted R²)."""
        scores = []

        for y_true, y_pred in zip(targets, predicted):
            # Rescale to original range for meaningful R² calculation
            y_true_rescaled = y_true
            y_pred_rescaled = y_pred
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true_rescaled, y_pred_rescaled)
            adj_r2 = self.adjusted_r2(
                y_true_rescaled, y_pred_rescaled,
                y_true.shape[0], self.input_dim
            )

            scores.append([mse, rmse, mae, r2, adj_r2])
        return scores

    def compute_imputation_metrics(self, real: List[np.ndarray],
                                   imputed: List[np.ndarray],
                                   rescale_params: Dict[str, float]) -> List[List[float]]:
        """Compute imputation quality metrics."""
        return self.compute_regression_metrics(real, imputed, rescale_params)

    @staticmethod
    def compute_binary_metrics(targets: List[np.ndarray],
                               predicted: List[np.ndarray]) -> List[List[float]]:
        """Compute binary classification metrics (ROC-AUC, PR-AUC)."""
        scores = []
        for y_true, y_pred in zip(targets, predicted):
            fpr, tpr, thresholds = roc_curve(y_pred, y_true)
            auc_score = auc(fpr, tpr)
            pr_score = average_precision_score(y_pred, y_true)
            scores.append([np.round(np.mean(auc_score), 4),
                           np.round(np.mean(pr_score), 4)])
        return scores

    @staticmethod
    def find_best_threshold(predictions: np.ndarray,
                            y_true: np.ndarray) -> Tuple[float, float]:
        """Find the optimal threshold for maximizing F1-score."""
        delta, tmp = 0, [0, 0, 0]  # idx, cur, max
        for tmp[0] in tqdm(np.arange(0.1, 1.01, 0.01)):
            tmp[1] = f1_score(y_true, np.array(predictions) > tmp[0])
            if tmp[1] > tmp[2]:
                delta = tmp[0]
                tmp[2] = tmp[1]
        print('best threshold is {:.2f} with F1 score: {:.4f}'.format(delta, tmp[2]))
        return delta, tmp[2]


class EarlyStopping:
    """
    Flexible Early Stopping for dual-objective training (primary task + imputation).

    Saves THREE checkpoints:
      - <base>_best_primary<ext>     : best-ever primary metric (accuracy or MSE), regardless of imputation
      - <base>_best_imputation<ext>  : best-ever imputation MAE, regardless of primary
      - <base>_best_combined<ext>    : best normalized joint score — THIS is what drives
                                        the patience counter / early stopping decision.

    Why a combined score and not "either improves = save + reset counter":
    That scheme lets one metric (typically the easier-to-improve one) mask
    regression in the other, since the checkpoint gets overwritten and the
    patience counter reset on ANY single-objective improvement — even if the
    metric you actually care about got worse that epoch. The combined score
    requires genuine joint improvement (or a good trade) before resetting
    patience, while the two side checkpoints remain available if you later
    want the single-metric-best weights instead.

    Normalization: each metric is expressed as a ratio against its value at
    the first call (the "baseline" epoch), so MSE (~1e-2 scale) and MAE
    (~1e-1 scale) contribute proportionally to the combined score regardless
    of their raw magnitude. Classification accuracy is converted internally
    to (1 - accuracy) so every component is "lower is better" before summing.
    """

    def __init__(self,
                 task: Optional[bool] = None,   # True/truthy = classification, falsy = regression
                 save_path: str = 'best_model.pt',
                 patience: int = 50,
                 primary_threshold: float = 0.0001,
                 imp_threshold: float = 0.00001,
                 combined_threshold: float = 0.0001,
                 primary_weight: float = 0.5,
                 imp_weight: float = 0.5,
                 logger: Optional[Any] = None):
        """
        Args:
            task: truthy -> classification (maximize accuracy), falsy -> regression (minimize MSE)
            save_path: base path; three files are derived from this
                       (..._best_primary / ..._best_imputation / ..._best_combined)
            patience: epochs without COMBINED improvement before stopping
            primary_threshold: min improvement to save the primary-only checkpoint
            imp_threshold: min improvement to save the imputation-only checkpoint
            combined_threshold: min improvement in normalized combined score to
                                 save the combined checkpoint and reset patience
            primary_weight / imp_weight: relative weighting in the combined score.
                                          Doesn't need to sum to 1; only the ratio matters.
            logger: logger instance
        """
        self.task = task
        self.patience = patience
        self.primary_threshold = primary_threshold
        self.imp_threshold = imp_threshold
        self.combined_threshold = combined_threshold
        self.primary_weight = primary_weight
        self.imp_weight = imp_weight
        self.logger = logger

        base, ext = os.path.splitext(save_path)
        self.primary_save_path = f"{base}_best_primary{ext}"
        self.imp_save_path = f"{base}_best_imputation{ext}"
        self.combined_save_path = f"{base}_best_combined{ext}"

        if task:
            self.best_primary = -np.inf
            self.primary_mode = 'max'
            self.primary_name = 'Accuracy'
        else:
            self.best_primary = np.inf
            self.primary_mode = 'min'
            self.primary_name = 'MSE'

        self.best_imp = np.inf
        self.best_epoch_primary = -1
        self.best_epoch_imp = -1

        # Combined-score tracking
        self.baseline_primary_loss = None   # set at first call
        self.baseline_imp = None
        self.best_combined_score = np.inf
        self.best_epoch_combined = -1

        # Overall tracking (patience is keyed to combined score only)
        self.counter = 0
        self.best_info = {}
        self.improvement_history = []

    @staticmethod
    def _safe(text: str) -> str:
        return text.encode("ascii", "replace").decode()

    def _log(self, message: str) -> None:
        safe_message = self._safe(message)
        if self.logger:
            self.logger.info(safe_message)
        else:
            print(safe_message)

    def _check_primary_improvement(self, primary_metric: float) -> bool:
        if self.primary_mode == 'max':
            return primary_metric > self.best_primary + self.primary_threshold
        else:
            return primary_metric < self.best_primary - self.primary_threshold

    def _get_primary_improvement_str(self, primary_metric: float) -> str:
        if self.primary_mode == 'max':
            improvement = (primary_metric - self.best_primary) * 100
            return f"{self.primary_name}: {self.best_primary:.4f} -> {primary_metric:.4f} (+{improvement:.2f}%)"
        else:
            improvement = (self.best_primary - primary_metric)
            return f"{self.primary_name}: {self.best_primary:.6f} -> {primary_metric:.6f} (-{improvement:.6f})"

    def _primary_as_loss(self, primary_metric: float) -> float:
        """Convert primary metric to a 'lower is better' quantity for combining."""
        if self.primary_mode == 'max':
            return 1.0 - primary_metric   # accuracy -> error rate
        return primary_metric             # MSE is already loss-like

    def _combined_score(self, primary_metric: float, imputation_metric: float) -> float:
        primary_loss = self._primary_as_loss(primary_metric)

        if self.baseline_primary_loss is None:
            # First call: establish baseline. Guard against a zero baseline.
            self.baseline_primary_loss = primary_loss if primary_loss > 1e-12 else 1e-12
            self.baseline_imp = imputation_metric if imputation_metric > 1e-12 else 1e-12

        norm_primary = primary_loss / self.baseline_primary_loss
        norm_imp = imputation_metric / self.baseline_imp

        return self.primary_weight * norm_primary + self.imp_weight * norm_imp

    def __call__(self,
                 primary_metric: float,
                 imputation_metric: float,
                 model: torch.nn.Module,
                 epoch: Optional[int] = None,
                 extra_metrics: Optional[Dict[str, float]] = None) -> bool:
        epoch_num = epoch + 1 if epoch is not None else 0
        any_improvement = False

        # ================================================================
        # SIDE CHECKPOINT 1: best-ever primary metric
        # ================================================================
        primary_improved = self._check_primary_improvement(primary_metric)
        if primary_improved:
            any_improvement = True
            self._log(f"[OK] Epoch {epoch_num}: {self._get_primary_improvement_str(primary_metric)} "
                      f"-> saved {os.path.basename(self.primary_save_path)}")
            self.best_primary = primary_metric
            self.best_epoch_primary = epoch_num
            torch.save(model.state_dict(), self.primary_save_path)

        # ================================================================
        # SIDE CHECKPOINT 2: best-ever imputation MAE
        # ================================================================
        imp_improved = imputation_metric < self.best_imp - self.imp_threshold
        if imp_improved:
            any_improvement = True
            imp_improvement = self.best_imp - imputation_metric
            self._log(f"[OK] Epoch {epoch_num}: Imputation MAE {self.best_imp:.6f} -> {imputation_metric:.6f} "
                      f"(-{imp_improvement:.6f}) -> saved {os.path.basename(self.imp_save_path)}")
            self.best_imp = imputation_metric
            self.best_epoch_imp = epoch_num
            torch.save(model.state_dict(), self.imp_save_path)

        # ================================================================
        # MAIN CHECKPOINT: combined normalized score.
        # Saved ONLY when the combined score itself improves — this file
        # must always hold the genuine combined-best weights, matching
        # best_epoch_combined. Side improvements (primary/imp alone) do NOT
        # write here, even though they DO reset the patience counter below.
        # ================================================================
        combined_score = self._combined_score(primary_metric, imputation_metric)
        combined_improved = combined_score < self.best_combined_score - self.combined_threshold

        if combined_improved:
            any_improvement = True
            self.best_combined_score = combined_score
            self.best_epoch_combined = epoch_num
            torch.save(model.state_dict(), self.combined_save_path)

            reason_str = (f"Combined score: {combined_score:.6f} "
                          f"(primary={primary_metric:.6f}, imputation={imputation_metric:.6f})")
            self._log(f"[OK] Epoch {epoch_num}: Combined checkpoint improved -> "
                      f"saved {os.path.basename(self.combined_save_path)}")
            self._log(f"  {reason_str}")

            self.best_info = {
                "epoch": epoch_num,
                "task": self.task,
                "best_combined_score": round(self.best_combined_score, 6),
                f"best_{self.primary_name.lower()}": round(primary_metric, 6),
                f"best_{self.primary_name.lower()}_epoch_overall": self.best_epoch_primary,
                "best_imputation_mae": round(imputation_metric, 6),
                "best_imputation_mae_epoch_overall": self.best_epoch_imp,
                "primary_at_combined_best": round(primary_metric, 6),
                "imputation_at_combined_best": round(imputation_metric, 6),
                "saved_at": datetime.now().isoformat(),
                "reason": reason_str,
                "checkpoint_paths": {
                    "primary": self.primary_save_path,
                    "imputation": self.imp_save_path,
                    "combined": self.combined_save_path,
                },
            }
            if extra_metrics:
                for key, value in extra_metrics.items():
                    self.best_info[f"extra_{key}"] = round(value, 6) if isinstance(value, float) else value

            self.improvement_history.append({
                "epoch": epoch_num,
                "primary_metric": primary_metric,
                "imputation_metric": imputation_metric,
                "combined_score": combined_score,
            })

        # ================================================================
        # BOOKKEEPING: counter resets on ANY of the three improving — this
        # is the part from last turn that stays. Only the SAVE for the
        # combined file is gated back to combined_improved only.
        # ================================================================
        if any_improvement:
            self.counter = 0
        else:
            self.counter += 1
            gap = combined_score - self.best_combined_score
            self._log(
                f"[X] Epoch {epoch_num}: No improvement (primary/imp/combined) | "
                f"combined_score: {combined_score:.6f} (best: {self.best_combined_score:.6f}, gap: {gap:.6f}) | "
                f"Counter: {self.counter}/{self.patience}"
            )

        early_stop = self.counter >= self.patience
        if early_stop:
            self._log(f"[EARLY STOP] Best combined checkpoint at Epoch {self.best_epoch_combined}")
            self._log(f"  Best combined score: {self.best_combined_score:.6f}")
            self._log(f"  Best {self.primary_name} (own best, may be a different epoch): "
                      f"{self.best_primary:.6f} (Epoch {self.best_epoch_primary})")
            self._log(f"  Best Imputation MAE (own best, may be a different epoch): "
                      f"{self.best_imp:.6f} (Epoch {self.best_epoch_imp})")

        return early_stop
