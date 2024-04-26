from django.contrib import admin
from django.urls import path

from caseApp.views import(CaseViewSet, create, upload_file)
from chatLegal.views import(chat_constituition)
from searchCases.views import search_case

urlpatterns = [
    path('admin/', admin.site.urls),
    # path("cases", CaseView.as_view(), name="cases"),
    # path("cases", CaseView.as_view(), name="cases")

    # path("cases/", CaseViewSet., name="cases")
    path("cases", create, name="cases"),
    path('upload/', upload_file, name="upload_file"),
    path('chat_constituition/', chat_constituition, name="chat_constituition"),
    path('case_search/', search_case, name="search_case"),
]
