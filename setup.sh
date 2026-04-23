#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$PROJECT_ROOT/src"

echo ""
echo "Starting KATSum setup..."
echo ""

mkdir -p "$SRC_DIR/results"
mkdir -p "$SRC_DIR/checkpoints"
mkdir -p "$SRC_DIR/dataset"

echo "  src/results, src/checkpoints, src/dataset are ready."
echo ""

VENV_DIR="$PROJECT_ROOT/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "No .venv found, creating one now..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created at .venv"
else
    echo "Found existing .venv, skipping creation."
fi

echo "Activating .venv..."
source "$VENV_DIR/bin/activate"
echo ""

REQUIREMENTS="$PROJECT_ROOT/requirements.txt"

if [ ! -f "$REQUIREMENTS" ]; then
    echo "Could not find requirements.txt at $REQUIREMENTS, aborting."
    exit 1
fi

echo "Installing dependencies from requirements.txt, this might take a while..."
echo ""
pip install -r "$REQUIREMENTS"
echo ""
echo "Dependencies installed."
echo ""

echo "Downloading spaCy en_core_web_sm model..."
echo ""
python3 -m spacy download en_core_web_sm
echo ""
echo "spaCy model ready."
echo ""

ZIP_FILE="$PROJECT_ROOT/pubmed_with_triples_v.zip"

if [ ! -f "$ZIP_FILE" ]; then
    echo "Could not find pubmed_with_triples_v.zip in the project root."
    echo "Place the zip file at root here and re-run setup.sh."
    exit 1
fi

echo "Found pubmed_with_triples_v.zip, extracting into src/dataset..."
unzip -o "$ZIP_FILE" -d "$SRC_DIR/dataset"
echo "Dataset extracted."
echo ""

RUN_ONCE="$SRC_DIR/load_model_tok_emb.py"

if [ ! -f "$RUN_ONCE" ]; then
    echo "Could not find load_model_tok_emb.py in src/, aborting."
    exit 1
fi

echo "Running load_model_tok_emb.py..."
echo ""
cd "$SRC_DIR" && python3 load_model_tok_emb.py
echo ""
echo "load_model_tok_emb.py done."
echo ""

echo "Setup done. Activate your environment any time with:"
echo ""
echo "  source .venv/bin/activate"
echo ""


echo "Final check: evaluation environment configuration..."
echo ""

EVAL_DIR="$PROJECT_ROOT/evaluation"
ENV_FILE="$EVAL_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "WARNING: ./evaluation/.env file is missing."
    echo ""
    echo "Refer to ./evaluation/.env.example for the required format."
    echo ""
    echo "Also ensure you add your GEMINI_API_KEY."
    echo ""
else
    if ! grep -q "^GEMINI_API_KEY=" "$ENV_FILE"; then
        echo "WARNING: GEMINI_API_KEY is not set in ./evaluation/.env"
        echo ""
        echo "Please open the .env file and add:"
        echo "  GEMINI_API_KEY=your_api_key_here"
        echo ""
        echo "Refer to .env.example for guidance."
        echo ""
    else
        echo "Evaluation environment looks correctly configured."
    fi
fi

echo ""