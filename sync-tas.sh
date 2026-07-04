#!/bin/bash
# Sync the TAS paper with Overleaf.  Usage:
#   ./sync-tas.sh pull            # get Overleaf web edits before you work
#   ./sync-tas.sh push "message"  # send local edits up to Overleaf
D="$(cd "$(dirname "$0")" && pwd)/paper-overleaf-tas"
cd "$D" || { echo "no paper-overleaf-tas/"; exit 1; }
case "$1" in
  pull) git pull ;;
  push) git add -A && git commit -m "${2:-edits}" && git push ;;
  *) echo "usage: ./sync-tas.sh pull | push [message]"; git status -s ;;
esac
