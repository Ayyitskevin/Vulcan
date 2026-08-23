#!/usr/bin/env bash
# One-time registration of the usage reporter's Athena forge source
# (deploy/README.md, "Usage reporter" step 1) — run by the operator with an
# ADMIN-scoped Athena token:
#
#   ATHENA_ADMIN_TOKEN=<token> bash deploy/register-reporter-source.sh
#
# The returned secret is shown by Athena exactly once. This script never
# prints it: the response is piped straight into a patch of the reporter's
# env file (mode 600), and only the source name and secret prefix are
# echoed. Mint the admin token for this one call and revoke it afterwards.
#
# A 409 means the source name is already registered; its secret cannot be
# re-read (one-time display) — delete and re-register from the Athena admin
# cockpit if the secret is lost.
set -euo pipefail

ATHENA_BASE_URL="${ATHENA_BASE_URL:-http://100.125.80.91:8300}"
SOURCE_NAME="${ATHENA_FORGE_SOURCE:-vulcan}"
ENV_FILE="${USAGE_REPORTER_ENV:-/home/kevin-lee/deploy/vulcan-data/usage-reporter.env}"

: "${ATHENA_ADMIN_TOKEN:?set ATHENA_ADMIN_TOKEN to an admin-scoped Athena token}"
[ -f "$ENV_FILE" ] || {
  echo "env file $ENV_FILE missing — copy deploy/usage-reporter.env.example first" >&2
  exit 2
}

# Status captured separately so a refusal surfaces as its own error body —
# error bodies carry no secret; only a 201 body (which does) reaches python.
# NOTE the role/scope trap: Athena caps token scope by USER role, so this
# needs an admin-scope token on an admin-ROLE user. The cockpit's
# "onboard agent" flow mints member-role agent users whose admin-scoped
# tokens still 403 here — mint from your own admin account instead.
response="$(curl -sS -w $'\n%{http_code}' -X POST "$ATHENA_BASE_URL/event-sources" \
  -H "Authorization: Bearer $ATHENA_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"name\": \"$SOURCE_NAME\", \"kind\": \"github\"}")"
status="${response##*$'\n'}"
body="${response%$'\n'*}"
if [ "$status" != "201" ]; then
  echo "registration refused: HTTP $status — $body" >&2
  exit 1
fi
printf '%s' "$body" |
  ENV_FILE="$ENV_FILE" SOURCE_NAME="$SOURCE_NAME" python3 - <<'PY'
import json
import os
import re
import sys

response = json.load(sys.stdin)
secret = response.get("secret", "")
if not secret.startswith("evtsec_"):
    raise SystemExit(f"registration answered without a secret: keys {sorted(response)}")

env_file = os.environ["ENV_FILE"]
with open(env_file, encoding="utf-8") as handle:
    body = handle.read()
patched, count = re.subn(
    r"^ATHENA_FORGE_SECRET=.*$", f"ATHENA_FORGE_SECRET={secret}", body, count=1, flags=re.M
)
if count != 1:
    raise SystemExit(f"{env_file} has no ATHENA_FORGE_SECRET line to patch")
with open(env_file, "w", encoding="utf-8") as handle:
    handle.write(patched)
os.chmod(env_file, 0o600)
print(f"source {os.environ['SOURCE_NAME']!r} registered; secret installed in "
      f"{env_file} ({secret[:10]}…, never shown again)")
PY
