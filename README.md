# Brain Tumor Detection API: Django + TensorFlow + LangChain

Yeh project aapke Google Colab notebook ko production-style Django REST API mein convert karta hai. **TensorFlow/VGG16 image classifier prediction karta hai; LangChain optional explanation layer hai.** LangChain image classifier ka replacement nahi hai. Agar `LANGCHAIN_ANALYSIS_ENABLED=False` rahe, API deterministic fallback explanation return karegi aur OpenAI key ki zaroorat nahi hogi.

> **Medical disclaimer:** Yeh project educational/technical demo hai, medical diagnosis device nahi. Kisi bhi result par qualified radiologist ya doctor se original scan review karwana zaroori hai.

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

`requirements.txt` mein TensorFlow ka CPU/GPU package apne machine ke hisaab se select karein. Production mein `DJANGO_SECRET_KEY`, `DJANGO_DEBUG=False`, aur `DJANGO_ALLOWED_HOSTS` zaroor set karein.

## 3. Model train/export karein

Dataset ka layout is tarah hona chahiye. Folder names `.env` ke `CLASS_LABELS` order se match hone chahiye.

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

Aapke actual Colab dataset paths yeh hain:

```python
train_dir = '/content/drive/MyDrive/Brain_tumar_detection/brain_tumer_dataset/Training'
test_dir = '/content/drive/MyDrive/Brain_tumar_detection/brain_tumer_dataset/Testing'
```

Agar training Colab mein dobara karni ho, to `scripts/train_model.py` upload karke yeh command chala sakte hain:

```bash
python scripts/train_model.py \\
  --train-dir /content/drive/MyDrive/Brain_tumar_detection/brain_tumer_dataset/Training \\
  --test-dir /content/drive/MyDrive/Brain_tumar_detection/brain_tumer_dataset/Testing \\
  --output /content/drive/MyDrive/Brain_tumar_detection/model.keras \\
  --epochs 6 \\
  --batch-size 20
```

Aapke existing model ka exact Colab path hai:

```text
/content/drive/MyDrive/Brain_tumar_detection/model.h5
```

Existing model ke liye dobara training ki zaroorat nahi. Colab mein yeh cell chala kar file download karein:

```python
from google.colab import files
files.download('/content/drive/MyDrive/Brain_tumar_detection/model.h5')
```

Downloaded `model.h5` ko local project ke is exact path par rakhein:

```text
brain_tumor_api/models/model.h5
```

Local `.env` mein yeh values set honi chahiye:

```dotenv
MODEL_PATH=models/model.h5
CLASS_LABELS=notumor,glioma,meningioma,pituitary
```

Colab ka `/content/drive/...` path local Windows/Linux Django server mein directly work nahi karega; model ko download karke `models/` folder mein copy karna compulsory hai.

## 4. Server start karein

```bash
python manage.py runserver 127.0.0.1:8000
```

Health check: `GET http://127.0.0.1:8000/health/`.

Model status: `GET http://127.0.0.1:8000/api/model-status/`.

## 5. Prediction API

Endpoint: `POST /api/predict/`. Request `multipart/form-data` honi chahiye aur field ka naam `image` hona chahiye.

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

## 6. LangChain explanation enable karna

`.env` mein yeh values set karein:

```dotenv
LANGCHAIN_ANALYSIS_ENABLED=True
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-4o-mini
```

Server restart ke baad `explanation.provider` `langchain-openai` aa sakta hai. API key ko source code mein hard-code na karein. Explanation sirf model ke already-generated class/confidence ko safe language mein summarize karti hai; prediction ko change nahi karti.

## 7. Notebook se important corrections

Aapke original notebook mein `ImageEnhance.Brightness(...).enhance(...)` aur contrast ka returned image assign nahi kiya gaya tha, isliye augmentation effective nahi thi. Inference mein model ko RGB image ka consistent preprocessing chahiye. Is project ke training script mein augmentation layers aur VGG16 preprocessing model graph ke andar rakhe gaye hain, jabki API raw RGB pixels ko `[0, 255]` range mein resize karke model ko deti hai. Isse training aur serving behavior consistent rehta hai.

`Flatten()` ke badle `GlobalAveragePooling2D()` use kiya gaya hai, jisse classifier head chhota aur comparatively less overfit-prone hota hai. Agar aap exact original `.h5` model use kar rahe hain, to output classes aur `CLASS_LABELS` ka order bilkul same rakhein.

## 8. Production checklist

Development server ki jagah Gunicorn use karein:

```bash
gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers 2
```

Production deployment mein HTTPS, reverse proxy, upload-size limits, authentication/rate limiting, structured logging, model versioning, audit controls, and clinical validation add karein. Patient-identifiable images ko unnecessary persist na karein; current endpoint image ko disk par save nahi karta.

## 9. API ko frontend se call karna

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

## 10. Files ka purpose

| File | Purpose |
|---|---|
| `detector/services/predictor.py` | Model loading, image validation, resize, and prediction |
| `detector/services/chain.py` | Optional LangChain explanation chain |
| `detector/views.py` | REST input validation and HTTP responses |
| `scripts/train_model.py` | Reproducible training and evaluation |
| `config/settings.py` | Environment-based application configuration |


## 11. One-command setup

Linux/Ubuntu par ZIP extract karne ke baad:

```bash
cd brain_tumor_api
chmod +x setup.sh
./setup.sh
```

Windows par `setup_windows.bat` par double-click karein. Dono scripts dependencies, `.env`, database migrations, aur folders automatically prepare karte hain. **Sirf `models/model.h5` manually copy karna hoga**, kyunki yeh aapki private Google Drive file hai.



## Local VGG16 + VGG19 models

Place the trained model files in the Django project's `models/` directory:

```text
models/model.h5        # VGG16
models/vgg19_model.h5  # VGG19
```

The Django website at `/` sends the uploaded image to both models. The comparison endpoint is `POST /api/compare/`. If the VGG19 filename differs, set `VGG19_MODEL_PATH` in `.env`.
