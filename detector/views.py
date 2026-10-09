from django.conf import settings
from rest_framework import serializers, status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView
from .services.predictor import predict
from .services.local_models import predict_vgg16_vgg19
from .services.chain import explain_prediction

class ImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField()
    def validate_image(self, value):
        if value.size > settings.MAX_UPLOAD_MB * 1024 * 1024:
            raise serializers.ValidationError(f"Image must be <= {settings.MAX_UPLOAD_MB} MB.")
        return value

class PredictView(APIView):
    parser_classes = [MultiPartParser, FormParser]
    def post(self, request):
        serializer = ImageUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            result = predict(serializer.validated_data["image"])
            result["explanation"] = explain_prediction(result)
            return Response(result, status=status.HTTP_200_OK)
        except FileNotFoundError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except (ValueError, RuntimeError) as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

class ModelStatusView(APIView):
    def get(self, request):
        from pathlib import Path
        path = Path(settings.MODEL_PATH)
        return Response({"model_exists": path.exists(), "model_path": str(path), "class_labels": settings.CLASS_LABELS})

class CompareModelsView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        serializer = ImageUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            result = predict_vgg16_vgg19(serializer.validated_data["image"])
            return Response(result, status=status.HTTP_200_OK)
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
