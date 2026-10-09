#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
mkdir -p models media
python manage.py migrate

if [ ! -f models/model.h5 ]; then
  echo
  echo "SETUP COMPLETE, BUT MODEL IS MISSING."
  echo "Copy your Colab file to: $(pwd)/models/model.h5"
  echo "Then run: source .venv/bin/activate && python manage.py runserver 127.0.0.1:8000"
else
  echo "SETUP COMPLETE. Model found at models/model.h5"
  echo "Start API with: source .venv/bin/activate && python manage.py runserver 127.0.0.1:8000"
fi
