# Deploying Vulcan as a service

This directory holds the reference systemd unit for running Vulcan as a
supervised, boot-persistent service on the operator's own machine. Vulcan
stays loopback-only and single-user; the unit changes nothing about that —
it only owns process lifecycle (start on boot, restart on failure, logs to
the journal).

The layout below is the convention the shipped unit assumes. Adjust paths in
`vulcan.service` if you choose a different one.

## Layout

| What | Where | Notes |
| --- | --- | --- |
| Code | `~/deploy/vulcan` | git clone of this repo, always on `main` |
| State | `~/deploy/vulcan-data` | `vulcan.toml`, usage ledger, future `.env` |

Code and state are separate on purpose: the checkout is disposable (re-clone
any time), `vulcan-data` is the only thing worth backing up.

## First install

```bash
git clone git@github.com:Ayyitskevin/Vulcan.git ~/deploy/vulcan
cd ~/deploy/vulcan && uv sync --all-groups --locked
mkdir -p ~/deploy/vulcan-data
cp config/vulcan.example.toml ~/deploy/vulcan-data/vulcan.toml
# Edit vulcan-data/vulcan.toml for your providers and aliases.
sudo cp deploy/vulcan.service /etc/systemd/system/vulcan.service
sudo systemctl daemon-reload
sudo systemctl enable --now vulcan
curl -fsS http://127.0.0.1:8140/healthz
```

## Update flow

```bash
bash ~/deploy/vulcan/deploy/update.sh
```

The script is the flow below made mechanical — use it rather than typing the
steps, because the one step that must never be skipped is easy to skip: the
deployed code validates the live config (`vulcan check`) BEFORE the restart,
so a TOML key that landed ahead of the code that knows it fails the update
with the old service still running, instead of taking the service down.

```bash
cd ~/deploy/vulcan
git pull --ff-only
uv sync --all-groups --locked
uv run vulcan check --config ~/deploy/vulcan-data/vulcan.toml
sudo systemctl restart vulcan
curl -fsS http://127.0.0.1:8140/healthz
```

The pull must be fast-forward only. A non-fast-forward pull means someone
wrote to the deploy checkout directly — stop and reconcile, do not force.
Config edits follow the same rule as everywhere else: take a timestamped
backup of `vulcan-data/vulcan.toml` first, then `systemctl restart vulcan`.
There is no config reload signal on purpose; the safe sequence for a config
change is `uv run vulcan check --config vulcan-data/vulcan.toml` first
(validates without starting a listener), then the restart.

## Watchdog (Type=notify)

The shipped unit runs `Type=notify` with `WatchdogSec=30`, paired with the
app's built-in sd_notify heartbeat (`src/vulcan/notify.py` — raw datagrams,
no dependency). `READY=1` at lifespan start gates "active" on the app
actually serving; `WATCHDOG=1` heartbeats run on the uvicorn event loop at
half the watchdog interval, so a wedged-but-alive process (the failure mode
`Restart=on-failure` cannot see) stops beating and systemd kills and
restarts it. Verify after install:

```bash
systemctl show vulcan -p WatchdogUSec   # expect 30s
systemctl status vulcan                 # expect "active (running)" post-READY
```

**Recovery drill** — run once after wiring the watchdog, and after any change
to the unit or notify code, so the kill-and-restart path is proven rather
than assumed (a watchdog that has never fired is a claim, not a mechanism):

```bash
kill -STOP "$(systemctl show vulcan -p MainPID --value)"  # simulate a wedge
journalctl -u vulcan -f          # expect "Watchdog timeout" then a restart
# systemd pairs the kill signal with SIGCONT, so even a stopped process
# dies promptly — expect recovery in about WatchdogSec + RestartSec
# (35s observed on the 2026-08-23 drill: timeout → SIGABRT → restart,
# ledger replayed 34/34 with zero skips).
curl -fsS http://127.0.0.1:8140/healthz   # recovered
curl -s http://127.0.0.1:8140/v1/usage | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["ledger"])'  # replay happened
```

Without `NOTIFY_SOCKET` in the environment (a plain shell, dev, CI) the
heartbeat is a complete no-op, so nothing else changes.

## Logs

```bash
journalctl -u vulcan -f
```

Vulcan logs content-safe structured JSON (fixed event names, recursive
redaction, never prompt or response text). The journal is the intended place
for them; no log files are written.

## Backups and state

`vulcan-data` is the only state. If `[usage] ledger_path` is configured, the
ledger is append-only JSONL and survives restarts by replay at startup; a
failed append never fails a completed request, but a lost ledger file means
lost history — include `vulcan-data` in whatever backup covers the machine.
One gateway process per ledger file.

## Usage reporter (daily digest to Athena)

`vulcan-usage-reporter.service` + `.timer` run `scripts/usage_reporter.py`
once a day (06:17 local, off the hour on purpose): read `/v1/usage`, diff
against the previous snapshot in `vulcan-data/usage-reporter-state.json`, and
post a compact digest — request/token counts per seat and per alias, budget
headroom, ledger honesty counters — into Athena's forge ingest. Content-safe
by construction: counters and labels only, which is all `/v1/usage` has.

Athena's forge speaks one dialect (`github`) and lands a delivery only where
it names a real issue key, so the digest arrives as imported history
(`forge_commit` rows, summary ≤ 200 chars each) on one standing issue.
`/v1/usage` is cumulative, so daily numbers are a snapshot diff: the first
run posts `[baseline]`, a counter regression or scope change posts `[reset]`
with cumulative values, and a missed day just stretches the next window
(`[delta 48h]`). One attempt per tick, no retries; any failure (Vulcan or
Athena unreachable, signature refused, delivery landed nowhere) exits
non-zero with one journald line. The reporter can never affect the `vulcan`
service — it shares nothing but the state directory.

One-time setup (operator):

```bash
# 1. Register the forge source (admin-scoped token required). The secret is
#    shown ONCE; losing it means re-registering.
curl -fsS -X POST http://100.125.80.91:8300/event-sources \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{"name": "vulcan", "kind": "github"}'
# → 201 {..., "secret": "evtsec_…"}

# 2. Create the standing issue the digest lands on (any project), e.g.
#    "Vulcan daily usage digest" → VUL-1.

# 3. Configure the reporter (0600, untracked — the secret lives here only).
cp deploy/usage-reporter.env.example ~/deploy/vulcan-data/usage-reporter.env
chmod 600 ~/deploy/vulcan-data/usage-reporter.env
# Edit: ATHENA_FORGE_SOURCE=vulcan, ATHENA_FORGE_SECRET=evtsec_…, ATHENA_ISSUE_KEY=VUL-1

# 4. Install the units and enable the timer (not the service).
sudo cp deploy/vulcan-usage-reporter.service deploy/vulcan-usage-reporter.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vulcan-usage-reporter.timer

# 5. Prove it end to end: the first manual run posts the baseline.
systemctl start vulcan-usage-reporter.service
journalctl -u vulcan-usage-reporter -n 5 --no-pager   # expect "baseline digest posted — N rows"
systemctl list-timers vulcan-usage-reporter.timer --no-pager
```

The service stays disabled; the timer owns the schedule. A manual
`systemctl start` any time posts a digest on demand.

Known quirk: Athena extracts issue-key candidates from message text, so a
seat or alias name ending in `-<digits>` (say `llama-3`) is a candidate key.
It lands only if a matching project/issue actually exists — avoid such names
if a colliding project key is ever created.

## Hosted BYOK keys

When hosted providers are enabled, their keys go in
`~/deploy/vulcan-data/.env` (mode `600`, untracked) as the variable names the
config's `api_key_env` fields reference, wired to the unit via the shipped
**drop-in** — never by editing the installed unit file:

```bash
sudo mkdir -p /etc/systemd/system/vulcan.service.d
sudo cp deploy/vulcan.service.d/local.conf /etc/systemd/system/vulcan.service.d/
sudo systemctl daemon-reload && sudo systemctl restart vulcan
```

Why the drop-in is mandatory: every unit update replaces
`/etc/systemd/system/vulcan.service` wholesale, so an edit made there — like
uncommenting the `EnvironmentFile` line — silently reverts on the next
`sudo cp`. Exactly that happened on 2026-08-23: the `Type=notify` unit flip
reverted the hosted-keys line and every BYOK alias failed with
`missing_credential` until the drop-in restored it. Drop-ins survive unit
replacement. Keys never go in the TOML, the unit, or the repo.
`vulcan check --config … --verify-credentials` confirms each key works
without printing values (run it with the env file sourced — see the
`check` trap above).
