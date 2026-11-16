#!/usr/bin/env bash
# Prints the directory that holds the model folders (DIR/<NAME>): $LM_MODELS_DIR when set,
# else $HOME/models (on the system SSD). install.sh, download-model.sh and doctor.sh default
# to it. A second disk for models is an explicit choice: set LM_MODELS_DIR to its mount.
set -euo pipefail
echo "${LM_MODELS_DIR:-$HOME/models}"
