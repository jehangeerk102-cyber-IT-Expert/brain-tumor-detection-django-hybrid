from django.urls import path
from .views import CompareModelsView, ModelStatusView, PredictView

urlpatterns = [
    path("predict/", PredictView.as_view(), name="predict"),
    path("model-status/", ModelStatusView.as_view(), name="model-status"),
    path("compare/", CompareModelsView.as_view(), name="compare-models"),
]
