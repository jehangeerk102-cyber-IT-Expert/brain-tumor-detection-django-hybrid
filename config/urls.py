from django.http import JsonResponse
from django.shortcuts import render
from django.urls import include, path


def api_home(request):
    return render(request, "detector/index.html")


urlpatterns = [
    path("", api_home, name="api-home"),
    path("api/", include("detector.urls")),
    path("health/", lambda request: JsonResponse({"status": "ok"})),
]
