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
# The unit's env file is sourced first so check sees the environment the
# service runs with; plain `check` still on purpose (no network probes —
# --verify-credentials is an explicit operator action, not an update step).
set -euo pipefail

DEPLOY_DIR="${VULCAN_DEPLOY_DIR:-/home/kevin-lee/deploy/vulcan}"
CONFIG="${VULCAN_CONFIG:-/home/kevin-lee/deploy/vulcan-data/vulcan.toml}"
HEALTHZ="${VULCAN_HEALTHZ:-http://127.0.0.1:8140/healthz}"
ENV_FILE="${VULCAN_ENV_FILE:-/home/kevin-lee/deploy/vulcan-data/.env}"

cd "$DEPLOY_DIR"

# Record the running revision before touching anything: if the new code
# passes check but fails at runtime, this is the rollback target.
PREV_SHA="$(git rev-parse --short HEAD)"
# A non-fast-forward pull means someone wrote to the deploy checkout
# directly — stop and reconcile, do not force (fails loudly here).
git pull --ff-only
uv sync --all-groups --locked
# Load the unit's hosted-keys env file when present so check sees the same
# environment the service runs with. check exits 1 for credentials missing
# from THIS shell — real information, but not a config failure — and 2 for
# a config the deployed code cannot parse, which is the failure this script
# exists to catch. Exit code alone is not enough: an unhandled crash in the
# new code also exits 1, so the missing-credentials pass requires check's
# stdout to carry its well-formed "config":"valid" report. A crash prints a
# traceback instead and aborts the update with the old service untouched.
if [ -r "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi
CHECK_RC=0
CHECK_OUT="$(uv run vulcan check --config "$CONFIG")" || CHECK_RC=$?
printf '%s\n' "$CHECK_OUT"
if [ "$CHECK_RC" -ne 0 ]; then
  if [ "$CHECK_RC" -eq 1 ] && printf '%s' "$CHECK_OUT" | grep -qF '"config":"valid"'; then
    echo "check: credentials missing from this shell (exit 1) — not a config failure, continuing"
  else
    echo "check failed (exit $CHECK_RC): config invalid or check crashed — aborting; service left untouched" >&2
    exit 1
  fi
fi
sudo systemctl restart vulcan
# Type=notify: restart returns only after the service announced READY=1.
curl -fsS "$HEALTHZ" > /dev/null
echo "vulcan updated to $(git rev-parse --short HEAD) and healthy (previous: $PREV_SHA)"
echo "rollback if it fails later: git reset --hard $PREV_SHA && uv sync --all-groups --locked && sudo systemctl restart vulcan"
echo "optional gate command 5: uv run python scripts/live_check.py --config $CONFIG"
