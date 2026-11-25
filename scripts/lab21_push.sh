#!/usr/bin/env bash
# 本仓库惯用的 push：带 credential.helper（Windows 的 GCM），失败时打印诊断。
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GIT_TERMINAL_PROMPT=0
git -c 'credential.helper=!"/mnt/c/Program Files/Git/mingw64/bin/git-credential-manager.exe"' push origin main
git rev-parse HEAD origin/main
