"""Model quality gates -- the single source of truth for promotion thresholds.

These thresholds are enforced in two places: the ``CheckModelQuality``
condition step in the SageMaker pipeline, and the pre-approval check in
``approve_model.py``. Defining them here keeps the two from drifting apart --
if the pipeline registers a model, the approval script must not then warn that
the same model fails the bar.

Why precision is set where it is
--------------------------------
The detector is an Isolation Forest fit on normal traffic only. Its decision
cutoff is chosen at training time from held-out labelled data (the threshold
that maximises F1), and stored in ``feature_manifest.json`` as
``anomaly_threshold``. Evaluation and serving both read that value, so the
metrics below describe the model as it is actually served.

Earlier versions used ``contamination=0.10`` as the effective cutoff. That
flags a fixed 10% of every batch regardless of the score distribution, which
manufactured roughly 90 false positives on 900 clean rows and held precision
near 0.48. Calibrating the threshold lifted precision to about 0.80 at 0.89
recall, with no change to the model itself.

Gates are set below the measured values with enough headroom to absorb normal
retraining variance, while still failing a genuine regression.
"""

from __future__ import annotations

# Minimum share of real attacks the model must catch. The primary gate.
MIN_RECALL = 0.70

# Minimum share of alerts that must be true attacks. Measured ~0.80 with the
# calibrated cutoff. Set at 0.70 so an uncalibrated model -- one that fell back
# to the contamination boundary and dropped to ~0.48 -- fails the build.
MIN_PRECISION = 0.70

# Ranking quality across all thresholds. Independent of the chosen operating
# point, so this is the most honest single measure of whether the model works.
MIN_PR_AUC = 0.65

# Ordered for display; the pipeline builds one condition per entry.
QUALITY_GATES: dict[str, float] = {
    "recall": MIN_RECALL,
    "precision": MIN_PRECISION,
    "pr_auc": MIN_PR_AUC,
}


def evaluate_gates(metrics: dict[str, float]) -> tuple[bool, list[str]]:
    """Check measured metrics against the gates.

    Returns whether every gate passed, plus one human-readable line per gate.
    """
    lines: list[str] = []
    passed = True

    for name, threshold in QUALITY_GATES.items():
        value = float(metrics.get(name, 0.0))
        ok = value >= threshold
        passed = passed and ok
        lines.append(
            f"{name:<10} {value:.4f} (required >= {threshold:.2f}) {'PASS' if ok else 'FAIL'}"
        )

    return passed, lines
