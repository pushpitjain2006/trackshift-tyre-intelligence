"""
Model diagnostics checking.

Validates MCMC sampling quality: R-hat convergence, effective sample size,
divergences. Triggers fallback if diagnostics are unhealthy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticsSummary:
    """Summary of MCMC sampling diagnostics."""
    max_rhat: float = 1.0
    min_ess: float = 0.0
    divergences: int = 0
    is_healthy: bool = True
    warnings: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        self._evaluate()

    def _evaluate(self):
        self.is_healthy = True
        if self.max_rhat > 1.01:
            self.is_healthy = False
            self.warnings.append(f"R-hat too high: {self.max_rhat:.3f} (threshold: 1.01)")
        if self.min_ess < 100:
            self.warnings.append(f"Low effective sample size: {self.min_ess:.0f}")
            if self.min_ess < 50:
                self.is_healthy = False
        if self.divergences > 0:
            self.warnings.append(f"Divergent transitions: {self.divergences}")
            if self.divergences > 10:
                self.is_healthy = False


def check_diagnostics(diagnostics_dict: dict) -> DiagnosticsSummary:
    """Evaluate diagnostics from a model result dictionary.

    Parameters
    ----------
    diagnostics_dict : dict
        From BayesianResult.diagnostics.

    Returns
    -------
    DiagnosticsSummary
    """
    return DiagnosticsSummary(
        max_rhat=diagnostics_dict.get("max_rhat", 1.0),
        min_ess=diagnostics_dict.get("min_ess", 0.0),
        divergences=diagnostics_dict.get("divergences", 0),
    )
