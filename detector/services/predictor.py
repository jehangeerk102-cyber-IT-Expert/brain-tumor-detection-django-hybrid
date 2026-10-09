from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
from django.conf import settings

IMAGE_SIZE = (128, 128)

@lru_cache(maxsize=1)
def get_model():
    model_path = Path(settings.MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train/export the model first.")
    from tensorflow.keras.models import load_model
    return load_model(model_path, compile=False)

def preprocess_image(uploaded_file):
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc
    image = image.resize(IMAGE_SIZE)
    # Matches the notebook's inference behavior: RGB float pixels in [0, 255].
    return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0), image

def predict(uploaded_file):
    batch, image = preprocess_image(uploaded_file)
    probabilities = np.asarray(get_model().predict(batch, verbose=0))[0]
    index = int(np.argmax(probabilities))
    labels = settings.CLASS_LABELS
    if index >= len(labels):
        raise RuntimeError("Number of configured class labels does not match the model output.")
    label = labels[index]
    return {
        "class_index": index,
        "class_label": label,
        "result": "No Tumor" if label.lower() == "notumor" else f"Tumor: {label}",
        "confidence": round(float(probabilities[index]), 6),
        "probabilities": {labels[i]: round(float(p), 6) for i, p in enumerate(probabilities[:len(labels)])},
        "image_size": {"width": image.width, "height": image.height},
    }
