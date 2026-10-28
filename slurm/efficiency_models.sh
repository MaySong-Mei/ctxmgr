#!/bin/bash
# Read-only reuse of group-readable, pinned team checkpoints. No credentials or writes to that cache.
set -euo pipefail
case "$1" in
  2B) rev=15852e8c16360a2fea060d615a32b45270f8a8fc ;;
  4B) rev=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a ;;
  9B) rev=c202236235762e1c871ad0ccb60c8ee5ba337b9a ;;
  27B) rev=fc05daec18b0a78c049392ed2e771dde82bdf654 ;;
  *) exit 2 ;;
esac
: "${TEAM_MODEL_CACHE:?Set TEAM_MODEL_CACHE to the readable team hub directory}"
printf '%s/models--Qwen--Qwen3.5-%s/snapshots/%s\n' "$TEAM_MODEL_CACHE" "$1" "$rev"
