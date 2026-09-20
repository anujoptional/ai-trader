#!/usr/bin/env bash
# Sync this repository with its GitHub remote using a personal access token.
#
# The token is read from .env at invocation time and handed to git through
# GIT_ASKPASS, which git calls only when it actually needs a credential. It is
# never written into .git/config, never embedded in a remote URL, and never
# handed to a credential helper for storage, so no path through this script can
# leave a token behind on disk or in a commit.
#
# Usage: scripts/sync.sh [extra git push arguments]

set -euo pipefail

# Git re-invokes this script as its askpass helper, passing the prompt text as
# the first argument. The marker variable is how that re-entry is told apart
# from a normal run; it is set only in the child environment exported below.
if [ -n "${AI_TRADER_ASKPASS:-}" ]; then
	case "${1:-}" in
	Username*) printf '%s\n' "${AI_TRADER_GIT_USER}" ;;
	*) printf '%s\n' "${AI_TRADER_GIT_TOKEN}" ;;
	esac
	exit 0
fi

PLACEHOLDER_TOKEN="replace-with-your-github-pat"

script_path="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
repository_root="$(git rev-parse --show-toplevel)"
env_file="${repository_root}/.env"

if [ ! -f "${env_file}" ]; then
	echo "No .env file found. Copy .env.example to .env first." >&2
	exit 2
fi

# Read one key the way python-dotenv does, so a line means the same thing here
# as it does to src/ai_trader/config.py: an optional "export " prefix and one
# layer of surrounding quotes are both accepted. A parser that took them
# literally would hand git a token wrapped in quote characters and report the
# resulting 403 as a bad token. The trailing carriage return strip is for .env
# files saved with CRLF endings. Values are assigned and never echoed.
read_env_value() {
	sed -n "s/^[[:space:]]*\(export[[:space:]]\{1,\}\)\{0,1\}$1=//p" "${env_file}" |
		head -n 1 |
		tr -d '\r' |
		sed -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'\$/\1/"
}

AI_TRADER_GIT_TOKEN="$(read_env_value GITHUB_PAT)"
AI_TRADER_GIT_USER="$(read_env_value GITHUB_USERNAME)"

if [ -z "${AI_TRADER_GIT_TOKEN}" ]; then
	echo "GITHUB_PAT is empty in .env." >&2
	exit 2
fi

if [ "${AI_TRADER_GIT_TOKEN}" = "${PLACEHOLDER_TOKEN}" ]; then
	echo "GITHUB_PAT in .env is still the placeholder. Replace it with a real token." >&2
	exit 2
fi

# GitHub ignores the username when the password is a token, so any non-empty
# value works and this one is only a fallback for an unset GITHUB_USERNAME.
AI_TRADER_GIT_USER="${AI_TRADER_GIT_USER:-x-access-token}"

if ! branch="$(git -C "${repository_root}" symbolic-ref --quiet --short HEAD)"; then
	echo "HEAD is detached; check out a branch before syncing." >&2
	exit 2
fi

remote_url="$(git -C "${repository_root}" remote get-url origin)"
# The prefix is matched whole rather than by host substring. A pattern like
# https://*github.com/* would also accept https://github.com@evil.com/, whose
# real host is evil.com and whose userinfo merely reads like GitHub; requiring
# the literal prefix rejects every credential-bearing URL of that shape. Only a
# remote that is unambiguously GitHub is ever offered the token.
case "${remote_url}" in
https://github.com/?*) ;;
*)
	echo "origin is not an https://github.com/ remote; refusing to send it the token." >&2
	exit 2
	;;
esac

export AI_TRADER_ASKPASS=1
export AI_TRADER_GIT_TOKEN
export AI_TRADER_GIT_USER
export GIT_ASKPASS="${script_path}"
# Fail instead of blocking on an interactive prompt if the token is rejected.
export GIT_TERMINAL_PROMPT=0

# An empty credential.helper switches off every configured helper for these two
# commands alone, so the token authenticates once and is not cached afterwards.
#
# Both commands name origin and the branch explicitly. That is what makes the
# check above binding rather than decorative: a bare "git pull" or "git push"
# resolves its remote from branch.<name>.remote, so a branch configured to track
# some other remote would be handed the token despite origin validating clean.
# Naming the remote on the command line also outranks a --repo= in the forwarded
# arguments, which is why those can be forwarded safely.
git -C "${repository_root}" -c credential.helper= pull --rebase --autostash origin "${branch}"
git -C "${repository_root}" -c credential.helper= push origin "${branch}" "$@"
