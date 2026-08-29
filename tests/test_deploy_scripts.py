"""Quality gates for the deploy shell scripts (deploy/*.sh).

The last production bugs shipped in these scripts, which had zero test
coverage. Two gates live here:

1. A syntax gate: every tracked shell script must pass ``bash -n``.
2. A behavioral gate for the config-check block in ``deploy/update.sh``:
   ``update.sh`` itself does ``git pull`` and ``systemctl restart``, so it
   cannot run end-to-end in CI. Instead the gate block is extracted from the
   script as text and executed under bash with a stubbed ``uv`` on PATH. The
   test reads the real file, so weakening or deleting the gate fails here.
"""

import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UPDATE_SH = REPO_ROOT / "deploy" / "update.sh"

# Sentinels printed by the harness around the extracted gate block.
CONTINUED = "GATE_CONTINUED"

VALID_REPORT = '{"config":"valid","credentials":"missing"}'


def _tracked_shell_scripts() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.sh"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [REPO_ROOT / line for line in out.stdout.splitlines()]


def _extract_gate_block() -> str:
    """Extract the check-gate block from deploy/update.sh, verbatim.

    The block is everything from ``CHECK_RC=0`` through the end of the gate's
    if-statement (just before the restart step). Both markers must exist —
    if someone deletes the gate or moves the restart before it, this raises
    and every gate test fails.
    """
    text = UPDATE_SH.read_text(encoding="utf-8")
    start = text.find("CHECK_RC=0")
    end = text.find("sudo systemctl restart")
    assert start != -1, "gate block start (CHECK_RC=0) not found in update.sh"
    assert end != -1, "restart step not found in update.sh"
    assert start < end, "gate block must come before the restart step"
    block = text[start:end]
    # Structural assertions: a weakened gate (no exit-code check, no
    # valid-config exception, no abort) must fail even if extraction worked.
    for needle in (
        "CHECK_RC=$?",
        '[ "$CHECK_RC" -ne 0 ]',
        '[ "$CHECK_RC" -eq 1 ]',
        '"config":"valid"',
        "exit 1",
    ):
        assert needle in block, f"gate block in update.sh is missing: {needle!r}"
    return block


def _run_gate(stub_rc: int, stub_out: str) -> subprocess.CompletedProcess[str]:
    """Run the extracted gate block under bash with a stubbed ``uv`` binary."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        bindir = Path(tmp)
        stub = bindir / "uv"
        stub.write_text(
            "#!/usr/bin/env bash\n"
            "# Stub for `uv run vulcan check`: emit canned stdout, exit canned rc.\n"
            'printf "%s\\n" "${STUB_OUT}"\n'
            'exit "${STUB_RC}"\n',
            encoding="utf-8",
        )
        stub.chmod(0o755)

        harness = (
            "set -euo pipefail\n"
            'CONFIG="/nonexistent/vulcan.toml"\n'
            f"{_extract_gate_block()}\n"
            f'echo "{CONTINUED}"\n'
        )
        env = dict(os.environ)
        env["PATH"] = f"{bindir}{os.pathsep}{env['PATH']}"
        env["STUB_RC"] = str(stub_rc)
        env["STUB_OUT"] = stub_out
        return subprocess.run(
            ["bash", "-c", harness],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )


# --- syntax gate over every shell script in the repo -----------------------


def test_all_shell_scripts_pass_bash_n() -> None:
    scripts = _tracked_shell_scripts()
    assert scripts, "no tracked *.sh files found — is git available in this checkout?"
    for script in scripts:
        result = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, f"bash -n failed for {script}: {result.stderr}"


# --- update.sh check-gate behavior -----------------------------------------


def test_gate_clean_exit_continues() -> None:
    """check exits 0 (config valid, credentials present) → update proceeds."""
    result = _run_gate(stub_rc=0, stub_out=VALID_REPORT)
    assert result.returncode == 0, result.stderr
    assert CONTINUED in result.stdout


def test_gate_missing_credentials_with_valid_report_continues() -> None:
    """check exits 1 but stdout carries the well-formed "config":"valid" report:
    credentials missing from the update shell is not a config failure."""
    result = _run_gate(stub_rc=1, stub_out=VALID_REPORT)
    assert result.returncode == 0, result.stderr
    assert CONTINUED in result.stdout
    assert "not a config failure" in result.stdout


def test_gate_exit_1_with_traceback_aborts() -> None:
    """check exits 1 with a crash traceback instead of the report → abort.
    Exit code alone must not be trusted: an unhandled crash also exits 1."""
    result = _run_gate(stub_rc=1, stub_out="Traceback (most recent call last):\nKeyError: 'x'")
    assert result.returncode == 1
    assert CONTINUED not in result.stdout
    assert "aborting" in result.stderr


def test_gate_exit_2_aborts() -> None:
    """check exits 2 (config the deployed code cannot parse) → abort; this is
    the failure the script exists to catch."""
    result = _run_gate(stub_rc=2, stub_out='{"config":"invalid","error":"extra key"}')
    assert result.returncode == 1
    assert CONTINUED not in result.stdout
    assert "aborting" in result.stderr


# --- update.sh rollback discipline ------------------------------------------


def test_prev_sha_recorded_before_git_pull() -> None:
    """The rollback target must be captured before anything is pulled."""
    text = UPDATE_SH.read_text(encoding="utf-8")
    prev = re.search(r'^PREV_SHA="\$\(git rev-parse', text, flags=re.M)
    pull = re.search(r"^git pull --ff-only", text, flags=re.M)
    assert prev, "PREV_SHA not recorded in update.sh"
    assert pull, "git pull --ff-only not found in update.sh"
    assert prev.start() < pull.start(), "PREV_SHA must be recorded before git pull"


def test_rollback_hint_printed() -> None:
    """After a successful update the script must tell the operator how to
    roll back to PREV_SHA."""
    text = UPDATE_SH.read_text(encoding="utf-8")
    assert re.search(r"git reset --hard \$PREV_SHA", text), (
        "update.sh no longer prints a rollback hint referencing PREV_SHA"
    )
