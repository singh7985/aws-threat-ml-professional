"""Dataset helper.

The complete deterministic dataset is already included in this package.
The existing project generator can create a smaller schema-compatible dataset:

python -m threat_ml.cli generate \
  --output data/raw/security_events.jsonl \
  --normal 4500 \
  --suspicious 500
"""

print(__doc__)
