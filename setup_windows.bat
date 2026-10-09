@echo off
setlocal
cd /d %~dp0

python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if not exist .env copy .env.example .env
if not exist models mkdir models
if not exist media mkdir media
python manage.py migrate

if not exist models\model.h5 (
  echo.
  echo SETUP COMPLETE, BUT MODEL IS MISSING.
  echo Copy your Colab model to: %CD%\models\model.h5
  echo Then run: .venv\Scripts\activate.bat ^&^& python manage.py runserver 127.0.0.1:8000
) else (
  echo SETUP COMPLETE. Model found.
  echo Start API with: .venv\Scripts\activate.bat ^&^& python manage.py runserver 127.0.0.1:8000
)
endlocal
