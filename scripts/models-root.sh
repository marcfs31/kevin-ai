#!/usr/bin/env bash
# Prints the directory that holds the model folders (DIR/<NAME>): $LM_MODELS_DIR when set;
# else /mnt/models when it is a directory this user can write (the models SSD, mounted and
# owned by the user; when the disk is not mounted the mount point is root's and read-only);
# else $HOME/models. install.sh, download-model.sh and doctor.sh all default to it.
set -euo pipefail
if [ -n "${LM_MODELS_DIR:-}" ]; then
  echo "$LM_MODELS_DIR"
  exit 0
fi
ssd="${LM_MODELS_MOUNT:-/mnt/models}"   # the tests point this at a temporary directory
if [ -d "$ssd" ] && [ -w "$ssd" ]; then echo "$ssd"; else echo "$HOME/models"; fi
