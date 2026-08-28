# Security policy

Vulcan is a public, local-first, single-user AI gateway that fronts real API
keys on your behalf. We take reports seriously and handle them privately.

## Reporting a vulnerability

Report privately through GitHub's [private vulnerability
reporting](https://github.com/Ayyitskevin/Vulcan/security/advisories/new) —
use **"New draft security advisory"** ("Report a vulnerability…"). Do not open
a public issue for suspected vulnerabilities; anything public is, by
definition, no longer reportable.

We acknowledge reports promptly and aim to ship or clearly plan a fix before
the advisory is published. If you are not comfortable using GitHub's flow,
contact the maintainer by other means and say "vulnerability report" in the
subject.

## Supported versions

The latest commit on `main` (and tagged releases that cut from it). There is
no backport track: Vulcan is single-operator infrastructure, and the
supported way to get a fix is to move the deployment forward.

## Scope — read before reporting

The gateway itself: request handling, routing, credential handling, logging,
configuration parsing, and the operator CLI.

The loopback-only listener and the `Host`-header allowlist are the
authentication model. That boundary is a documented design choice — see the
[Security and privacy
defaults](README.md#security-and-privacy-defaults) section in the README —
not a bug. In particular, these are **not** findings:

- No credentials, sessions, or multi-user isolation: the operator's machine
  is the trust boundary.
- The operator's API keys are only as secret as the shell, systemd unit, and
  `EnvironmentFile` that export them; Vulcan does not and cannot protect
  against compromise of that environment.
- `http` upstream URLs on loopback (documented allowance for local proxies).
- Absence of TLS on the local listener.

In scope, for example: prompt or upstream-body text leaking into responses,
errors, or logs; a request path that can select or construct an upstream URL;
a failure that violates the no-retry, no-fallback contract in a way that can
double-charge; a path that bypasses budgets or the in-flight bound; a startup
path that accepts invalid configuration and fails unsafely.

## What you can expect

- Private handling; no disclosure before the fix is available.
- A clear reason for any finding we judge out of scope, tied to the design
  record.
- Credit in the advisory if requested.
