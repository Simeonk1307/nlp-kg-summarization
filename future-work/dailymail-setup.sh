#!/bin/bash
set -euo pipefail

KAGGLE_LINK="https://www.kaggle.com/api/v1/datasets/download/gowrishankarp/newspaper-text-summarization-cnn-dailymail"
ZIP_FILE_PATH="./cnn-dailymail.zip"

BLUE="\e[1;34m"
GREEN="\e[1;32m"
RED="\e[1;31m"
NC="\e[0m"

confirm() {
    local prompt="$1"
    local skip_msg="$2"

    echo -e "\n${prompt}"
    echo -e "${BLUE}==> [Y|y] (default)\t${NC}[N|n] (skip)"
    echo -ne "${BLUE}==> ${NC}"

    local choice=""
    read -r -n 1 choice
    echo

    case "${choice:-Y}" in
        [Yy]) return 0 ;;
        [Nn]) echo -e "${GREEN}${skip_msg}${NC}"; return 1 ;;
        *)    echo -e "${RED}Invalid character '${choice}'. Aborting.${NC}"; exit 1 ;;
    esac
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo -e "${RED}$1 does not exist (download it first). Aborting.${NC}"
        exit 2
    fi
}

echo -e "${GREEN}This is a setup script which downloads the CNN-DailyMail dataset from Kaggle, unzips it to PWD and performs basic Python venv setup.${NC}"
echo -e "Kaggle link: ${BLUE}${KAGGLE_LINK}${NC}"
echo -e "Working dir: ${BLUE}${PWD}${NC}"
echo -e "Requirements: ${RED}curl, unzip, python3${NC}"

if confirm "Download the CNN-DailyMail zip from Kaggle? (No, if already downloaded)" "Skipping download."; then
    echo -e "${GREEN}Downloading zip file...${NC}"
    curl -L -o "${ZIP_FILE_PATH}" "${KAGGLE_LINK}"
fi

if confirm "Unzip ${ZIP_FILE_PATH}? (No, if already unzipped)" "Skipping unzip."; then
    require_file "${ZIP_FILE_PATH}"
    echo -e "${GREEN}Unzipping...${NC}"
    unzip -o "${ZIP_FILE_PATH}"
fi

if confirm "Python '.venv' setup? (No, if already installed)" "Skipping '.venv' setup."; then
    echo -e "${GREEN}Setting up python venv...${NC}"
    python3 -m venv .venv
    source .venv/bin/activate
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
    deactivate
fi


echo -e "\n${GREEN}Complete Setup done.${NC}"