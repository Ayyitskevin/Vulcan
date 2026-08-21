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
cd ~/deploy/vulcan
git pull --ff-only
uv sync --all-groups --locked
sudo systemctl restart vulcan
curl -fsS http://127.0.0.1:8140/healthz
```

The pull must be fast-forward only. A non-fast-forward pull means someone
wrote to the deploy checkout directly — stop and reconcile, do not force.
Config edits follow the same rule as everywhere else: take a timestamped
backup of `vulcan-data/vulcan.toml` first, then `systemctl restart vulcan`.

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

## Hosted BYOK keys

When hosted providers are enabled, their keys go in
`~/deploy/vulcan-data/.env` (mode `600`, untracked) as the variable names the
config's `api_key_env` fields reference, and the commented `EnvironmentFile`
line in the unit is uncommented. Keys never go in the TOML, the unit, or the
repo. `vulcan check --config … --verify-credentials` confirms each key works
without printing values.
