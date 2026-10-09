"""Local VGG16 and VGG19 inference for the website/API."""

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from django.conf import settings

IMAGE_SIZE = (128, 128)


@lru_cache(maxsize=2)
def _load_model(model_path):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    from tensorflow.keras.models import load_model
    return load_model(path, compile=False)


def _prepare(uploaded_file):
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc
    original_size = {"width": image.width, "height": image.height}
    image = image.resize(IMAGE_SIZE)
    batch = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
    return batch, original_size


def _one_prediction(batch, model_path, model_name):
    probabilities = np.asarray(_load_model(str(model_path)).predict(batch, verbose=0))[0]
    labels = settings.CLASS_LABELS
    if len(probabilities) != len(labels):
        raise RuntimeError(
            f"{model_name} output has {len(probabilities)} classes but CLASS_LABELS has {len(labels)}."
        )
    index = int(np.argmax(probabilities))
    label = labels[index]
    return {
        "model": model_name,
        "model_path": str(model_path),
        "class_index": index,
        "class_label": label,
        "result": "No Tumor" if label.lower() == "notumor" else f"Tumor: {label}",
        "confidence": round(float(probabilities[index]), 6),
        "probabilities": {labels[i]: round(float(p), 6) for i, p in enumerate(probabilities)},
    }


def predict_vgg16_vgg19(uploaded_file):
    batch, image_size = _prepare(uploaded_file)
    results = {}
    errors = {}
    for key, name, path in (
        ("vgg16", "VGG16", settings.VGG16_MODEL_PATH),
        ("vgg19", "VGG19", settings.VGG19_MODEL_PATH),
    ):
        try:
            results[key] = _one_prediction(batch, path, name)
        except (FileNotFoundError, RuntimeError) as exc:
            errors[key] = str(exc)
    if not results:
        raise RuntimeError("Neither VGG16 nor VGG19 model could predict: " + str(errors))
    return {"image_size": image_size, "models": results, "errors": errors}
