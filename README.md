# Brain Tumor Detection API: Django + TensorFlow + LangChain

This project converts your Google Colab notebook into a production-style Django REST API. **TensorFlow/VGG16 performs the image classification prediction; LangChain is an optional explanation layer.** LangChain is not a replacement for the image classifier. If `LANGCHAIN_ANALYSIS_ENABLED=False`, the API returns a deterministic fallback explanation and no OpenAI key is required.

> **Medical disclaimer:** This project is an educational/technical demo, not a medical diagnosis device. Any result should be reviewed with a qualified radiologist or doctor using the original scan.

## 1. Directory structure

```text
brain_tumor_api/
├── .env.example
├── .gitignore
├── requirements.txt
├── manage.py
├── README.md
├── config/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
├── detector/
│   ├── __init__.py
│   ├── apps.py
│   ├── urls.py
│   ├── views.py
│   └── services/
│       ├── __init__.py
│       ├── predictor.py
│       └── chain.py
├── models/
│   └── brain_tumor_vgg16.keras
├── media/
├── scripts/
│   └── train_model.py
└── db.sqlite3
```

## 2. Environment setup

```bash
cd /home/ubuntu/brain_tumor_api
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
python manage.py migrate
```

In `requirements.txt`, select the TensorFlow CPU/GPU package according to your machine. In production, be sure to set `DJANGO_SECRET_KEY`, `DJANGO_DEBUG=False`, and `DJANGO_ALLOWED_HOSTS`.

## 3. Train/export the model

The dataset layout should look like this. Folder names must match the `CLASS_LABELS` order in `.env`.

```text
brain_tumer_dataset/
├── Training/
│   ├── notumor/
│   ├── glioma/
│   ├── meningioma/
│   └── pituitary/
└── Testing/
    ├── notumor/
    ├── glioma/
    ├── meningioma/
    └── pituitary/
```

Your actual Colab dataset paths are:

```python
train_dir = '/content/drive/MyDrive/Brain_tumar_detection/brain_tumer_dataset/Training'
test_dir = '/content/drive/MyDrive/Brain_tumar_detection/brain_tumer_dataset/Testing'
```

If you want to retrain in Colab, upload `scripts/train_model.py` and run this command:

```bash
python scripts/train_model.py \
  --train-dir /content/drive/MyDrive/Brain_tumar_detection/brain_tumer_dataset/Training \
  --test-dir /content/drive/MyDrive/Brain_tumar_detection/brain_tumer_dataset/Testing \
  --output /content/drive/MyDrive/Brain_tumar_detection/model.keras \
  --epochs 6 \
  --batch-size 20
```

Your existing model's exact Colab path is:

```text
/content/drive/MyDrive/Brain_tumar_detection/model.h5
```

No retraining is needed for the existing model. Run this cell in Colab to download the file:

```python
from google.colab import files
files.download('/content/drive/MyDrive/Brain_tumar_detection/model.h5')
```

Place the downloaded `model.h5` at this exact path in the local project:

```text
brain_tumor_api/models/model.h5
```

Your local `.env` should have these values:

```dotenv
MODEL_PATH=models/model.h5
CLASS_LABELS=notumor,glioma,meningioma,pituitary
```

The Colab path `/content/drive/...` will not work directly in a local Windows/Linux Django server; downloading the model and copying it into the `models/` folder is mandatory.

## 4. Start the server

```bash
python manage.py runserver 127.0.0.1:8000
```

Health check: `GET http://127.0.0.1:8000/health/`.

Model status: `GET http://127.0.0.1:8000/api/model-status/`.

## 5. Prediction API

Endpoint: `POST /api/predict/`. The request must be `multipart/form-data` and the field name must be `image`.

```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -F "image=@/path/to/scan.jpg"
```

Example response:

```json
{
  "class_index": 1,
  "class_label": "glioma",
  "result": "Tumor: glioma",
  "confidence": 0.934512,
  "probabilities": {
    "notumor": 0.012,
    "glioma": 0.934512,
    "meningioma": 0.031,
    "pituitary": 0.022488
  },
  "image_size": {"width": 128, "height": 128},
  "explanation": {
    "summary": "AI prediction: Tumor: glioma with 93.45% confidence. This is not a medical diagnosis...",
    "provider": "fallback"
  }
}
```

Possible HTTP responses are `200` for a prediction, `400` for invalid image/input, `413`-style validation behavior for oversized uploads depending on the server layer, and `503` when the model file is missing.

## 6. Enabling LangChain explanation

Set these values in `.env`:

```dotenv
LANGCHAIN_ANALYSIS_ENABLED=True
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-4o-mini
```

After restarting the server, `explanation.provider` may become `langchain-openai`. Do not hard-code the API key in the source code. The explanation only summarizes the model's already-generated class/confidence in safe language; it does not change the prediction.

## 7. Important corrections from the notebook

In your original notebook, the image returned by `ImageEnhance.Brightness(...).enhance(...)` and contrast was not assigned, so the augmentation was not effective. Inference requires consistent preprocessing of the RGB image for the model. In this project's training script, the augmentation layers and VGG16 preprocessing are kept inside the model graph, while the API resizes raw RGB pixels in the `[0, 255]` range and passes them to the model. This keeps training and serving behavior consistent.

`GlobalAveragePooling2D()` is used instead of `Flatten()`, which makes the classifier head smaller and comparatively less overfit-prone. If you are using the exact original `.h5` model, keep the output classes and `CLASS_LABELS` order exactly the same.

## 8. Production checklist

Use Gunicorn instead of the development server:

```bash
gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers 2
```

In production deployment, add HTTPS, a reverse proxy, upload-size limits, authentication/rate limiting, structured logging, model versioning, audit controls, and clinical validation. Do not unnecessarily persist patient-identifiable images; the current endpoint does not save images to disk.

## 9. Calling the API from a frontend

```javascript
const form = new FormData();
form.append("image", fileInput.files[0]);
const response = await fetch("http://127.0.0.1:8000/api/predict/", {
  method: "POST",
  body: form,
});
const data = await response.json();
console.log(data.result, data.confidence);
```

## 10. Purpose of files

| File | Purpose |
|---|---|
| `detector/services/predictor.py` | Model loading, image validation, resize, and prediction |
| `detector/services/chain.py` | Optional LangChain explanation chain |
| `detector/views.py` | REST input validation and HTTP responses |
| `scripts/train_model.py` | Reproducible training and evaluation |
| `config/settings.py` | Environment-based application configuration |

## 11. One-command setup

On Linux/Ubuntu after extracting the ZIP:

```bash
cd brain_tumor_api
chmod +x setup.sh
./setup.sh
```

On Windows, double-click `setup_windows.bat`. Both scripts automatically prepare dependencies, `.env`, database migrations, and folders. **Only `models/model.h5` must be copied manually**, because it is your private Google Drive file.

## Local VGG16 + VGG19 models

Place the trained model files in the Django project's `models/` directory:

```text
models/model.h5        # VGG16
models/vgg19_model.h5  # VGG19
```

The Django website at `/` sends the uploaded image to both models. The comparison endpoint is `POST /api/compare/`. If the VGG19 filename differs, set `VGG19_MODEL_PATH` in `.env`.
