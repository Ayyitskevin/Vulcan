#!/usr/bin/env bash
# Mechanical update for the deployed gateway — the documented update flow
# (deploy/README.md) made executable so no step can be skipped or reordered.
#
# The step that must never be skipped is the third one: the DEPLOYED code
# validates the LIVE config before any restart. Config parsing is
# extra="forbid", so a TOML key that lands ahead of the code that knows it
# would otherwise take the service down at restart; here it fails the update
# instead, with the service still running the old code untouched.
#
# Plain `check` on purpose, not `check --verify-credentials`: hosted keys
# live only in the unit's EnvironmentFile, so credential verification from
# an operator shell reports false negatives (see README, "check" trap).
set -euo pipefail

DEPLOY_DIR="${VULCAN_DEPLOY_DIR:-/home/kevin-lee/deploy/vulcan}"
CONFIG="${VULCAN_CONFIG:-/home/kevin-lee/deploy/vulcan-data/vulcan.toml}"
HEALTHZ="${VULCAN_HEALTHZ:-http://127.0.0.1:8140/healthz}"

cd "$DEPLOY_DIR"

# A non-fast-forward pull means someone wrote to the deploy checkout
# directly — stop and reconcile, do not force (fails loudly here).
git pull --ff-only
uv sync --all-groups --locked
uv run vulcan check --config "$CONFIG"
sudo systemctl restart vulcan
# Type=notify: restart returns only after the service announced READY=1.
curl -fsS "$HEALTHZ" > /dev/null
echo "vulcan updated to $(git rev-parse --short HEAD) and healthy"
echo "optional gate command 5: uv run python scripts/live_check.py --config $CONFIG"
