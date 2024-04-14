"""
URL configuration for legalLensApi project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from caseApp.views import(CaseViewSet, chat_constituition, create, upload_file)

urlpatterns = [
    path('admin/', admin.site.urls),
    # path("cases", CaseView.as_view(), name="cases"),
    # path("cases", CaseView.as_view(), name="cases")

    # path("cases/", CaseViewSet., name="cases")
    path("cases", create, name="cases"),
    path('upload/', upload_file, name="upload_file"),
    path('chat_constituition/', chat_constituition, name="chat_constituition"),
]
