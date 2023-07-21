#!/bin/bash

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

function main() {
  trap exit SIGINT

  cd "${THIS_DIR}"

  # Download and prepare data.
  ./data/fetchData.sh

  if [[ -e ./data/rgb-images ]]; then
    echo "rgb-images directory found cached, skipping image extraction."
  else
    python3 data/extract_videos2jpgs.py "${THIS_DIR}/data"
  fi

  # Run task1.
  python3 task1.py
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
