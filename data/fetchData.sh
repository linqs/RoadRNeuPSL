#!/bin/bash

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

function main() {
    trap exit SIGINT

    cd "${THIS_DIR}"

    check_requirements

    fetch_videos_file
    fetch_annotations_file
    fetch_counts_file

    extract_videos_zip
}

function check_requirements() {
    local hasWget

    type wget > /dev/null 2> /dev/null
    hasWget=$?

    if [[ "${hasWget}" -ne 0 ]]; then
        echo 'ERROR: wget or curl required to download the jar.'
        exit 10
    fi
}

function fetch_videos_file() {
    if [[ -e "videos.zip" ]]; then
        echo "videos.zip file found cached, skipping download."
        return
    fi

    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1YQ9ap3o9pqbD0Pyei68rlaMDcRpUn-qz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1YQ9ap3o9pqbD0Pyei68rlaMDcRpUn-qz" -O videos.zip
    if [[ "$?" -ne 0 ]]; then
        echo "ERROR: Failed to download videos.zip."
        exit 30
    fi
}

function fetch_annotations_file() {
    if [[ -e "road_trainval_v1.0.json" ]]; then
        echo "road_trainval_v1.0.json file found cached, skipping download."
        return
    fi

    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1HAJpdS76TVK56Qvq1jXr-5hfFCXKHRZo' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1HAJpdS76TVK56Qvq1jXr-5hfFCXKHRZo" -O road_trainval_v1.0.json
    if [[ "$?" -ne 0 ]]; then
        echo "ERROR: Failed to download road_trainval_v1.0.json."
        exit 30
    fi
}

function fetch_counts_file() {
    if [[ -e "instance_counts.json" ]]; then
        echo "instance_counts.json file found cached, skipping download."
        return
    fi

    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1NfSoI1yVTA46YY7AwVIGRolAqtWfoa8V' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1NfSoI1yVTA46YY7AwVIGRolAqtWfoa8V" -O instance_counts.json
    if [[ "$?" -ne 0 ]]; then
        echo "ERROR: Failed to download instance_counts.json."
        exit 30
    fi
}


function extract_videos_zip() {
    if [[ -e "videos" ]]; then
        echo "Extracted videos zip found cached, skipping extract."
        return
    fi

    echo "Extracting the videos zip"
    unzip "videos.zip"
    if [[ "$?" -ne 0 ]]; then
        echo "ERROR: Failed to extract videos.zip."
        exit 40
    fi
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
