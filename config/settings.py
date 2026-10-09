from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-only-change-me")
DEBUG = os.getenv("DJANGO_DEBUG", "False").lower() == "true"
ALLOWED_HOSTS = [x.strip() for x in os.getenv("DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",") if x.strip()]
INSTALLED_APPS = ["django.contrib.admin", "django.contrib.auth", "django.contrib.contenttypes", "django.contrib.sessions", "django.contrib.messages", "django.contrib.staticfiles", "rest_framework", "detector"]
MIDDLEWARE = ["django.middleware.security.SecurityMiddleware", "whitenoise.middleware.WhiteNoiseMiddleware", "django.contrib.sessions.middleware.SessionMiddleware", "django.middleware.common.CommonMiddleware", "django.middleware.csrf.CsrfViewMiddleware", "django.contrib.auth.middleware.AuthenticationMiddleware", "django.contrib.messages.middleware.MessageMiddleware"]
ROOT_URLCONF = "config.urls"
TEMPLATES = [{"BACKEND": "django.template.backends.django.DjangoTemplates", "DIRS": [], "APP_DIRS": True, "OPTIONS": {"context_processors": ["django.template.context_processors.request", "django.contrib.auth.context_processors.auth", "django.contrib.messages.context_processors.messages"]}}]
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": BASE_DIR / "db.sqlite3"}}
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True
STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
MODEL_PATH = BASE_DIR / os.getenv("MODEL_PATH", "models/model.h5")
VGG16_MODEL_PATH = BASE_DIR / os.getenv("VGG16_MODEL_PATH", "models/model.h5")
VGG19_MODEL_PATH = BASE_DIR / os.getenv("VGG19_MODEL_PATH", "models/vgg19_model.h5")
CLASS_LABELS = [x.strip() for x in os.getenv("CLASS_LABELS", "notumor,glioma,meningioma,pituitary").split(",")]
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
LANGCHAIN_ANALYSIS_ENABLED = os.getenv("LANGCHAIN_ANALYSIS_ENABLED", "False").lower() == "true"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
REST_FRAMEWORK = {"DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"]}
