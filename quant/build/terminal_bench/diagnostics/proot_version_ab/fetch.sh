#!/usr/bin/env bash
# Official HTTPS downloads only. A failed request stops; no proxy/mirror bypass.
set -euo pipefail
diag="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$diag/downloads" "$diag/logs"
fetch() {
  local url="$1" name="$2"
  if [[ ! -f "$diag/downloads/$name" ]]; then
    timeout --kill-after=2 55 curl --fail --location --silent --show-error \
      --connect-timeout 12 --max-time 50 "$url" -o "$diag/downloads/$name.part" \
      >"$diag/logs/fetch-$name.stdout" 2>"$diag/logs/fetch-$name.stderr"
    mv "$diag/downloads/$name.part" "$diag/downloads/$name"
  fi
}
if [[ ! -d "$diag/source/.git" ]]; then
  timeout --kill-after=2 60 git clone --depth 1 --branch v5.5.0 \
    https://github.com/proot-me/proot.git "$diag/source" \
    >"$diag/logs/clone.stdout" 2>"$diag/logs/clone.stderr"
fi
if ! git -C "$diag/source" rev-parse --verify refs/tags/v5.4.1 >/dev/null 2>&1; then
  timeout --kill-after=2 50 git -C "$diag/source" fetch --depth 1 origin \
    refs/tags/v5.4.1:refs/tags/v5.4.1 >"$diag/logs/fetch-5.4.1.stdout" 2>"$diag/logs/fetch-5.4.1.stderr"
fi
fetch https://api.github.com/repos/proot-me/proot/releases/tags/v5.5.0 release.json
fetch https://github.com/proot-me/proot/releases/download/v5.5.0/proot proot
fetch https://github.com/proot-me/proot/releases/download/v5.5.0/SHA256SUMS SHA256SUMS
fetch https://api.github.com/repos/proot-me/proot/releases/tags/v5.4.1 release-5.4.1.json
fetch https://github.com/proot-me/proot/releases/download/v5.4.1/proot proot-5.4.1
fetch https://github.com/proot-me/proot/releases/download/v5.4.1/SHA256SUMS SHA256SUMS-5.4.1
chmod 755 "$diag/downloads/proot" "$diag/downloads/proot-5.4.1"
printf 'Official release files available; prepare.py performs the hash checks.\n'
