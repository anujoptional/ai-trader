#!/usr/bin/env bash
# Build the development environment this repository's gates expect.
#
# Creates .venv on the interpreter named in .python-version, installs the
# project in editable mode with its dev extras, and bounds every version by
# constraints.txt so a fresh machine gets the same package set the gates were
# last validated against rather than whatever PyPI is serving today.
#
# Safe to re-run: an existing .venv is reused and brought up to date.
#
# Usage: scripts/bootstrap.sh

set -euo pipefail

repository_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${repository_root}"

if ! command -v uv >/dev/null 2>&1; then
	echo "uv is not installed. See https://docs.astral.sh/uv/getting-started/" >&2
	exit 2
fi

# Git Bash on Windows runs this script against a venv laid out the Windows way.
case "$(uname -s)" in
MINGW* | MSYS* | CYGWIN*) venv_python=".venv/Scripts/python.exe" ;;
*) venv_python=".venv/bin/python" ;;
esac

# uv reads .python-version here, and fetches that interpreter if the machine
# does not already have it.
echo "==> Creating .venv"
uv venv

echo "==> Installing the project and its dev extras"
uv pip install \
	--python "${venv_python}" \
	--constraint constraints.txt \
	--editable ".[dev]"

# A working import is the cheapest proof that the editable install actually
# took, and it fails loudly here rather than as a confusing collection error on
# the first pytest run.
echo "==> Verifying the install"
"${venv_python}" -c "import ai_trader; print('ai_trader imports cleanly')"

if [ ! -f "${repository_root}/.env" ]; then
	# Deliberately not created for you. An .env full of empty values looks
	# configured and fails at the broker with a worse message than an absent
	# one, and this script has no business writing a credential file.
	echo
	echo "Note: no .env file. Broker-backed commands need one:"
	echo "      cp .env.example .env   # then fill it in"
fi

cat <<'EOF'

Done. The three gates are:

    .venv/Scripts/python.exe -m ruff format --check .   # or .venv/bin/python
    .venv/Scripts/python.exe -m ruff check .
    .venv/Scripts/python.exe -m pytest

EOF
