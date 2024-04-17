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
