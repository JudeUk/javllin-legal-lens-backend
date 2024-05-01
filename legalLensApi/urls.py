from django.contrib import admin
from django.urls import path

from caseApp.views import(CaseViewSet, create, upload_file)
from chatLegal.views import(chat_constituition)
# from searchCases.views import case_search
from searchCases.views import text_index_search_case

urlpatterns = [
    path('admin/', admin.site.urls),
    # path("cases", CaseView.as_view(), name="cases"),
    # path("cases", CaseView.as_view(), name="cases")

    # path("cases/", CaseViewSet., name="cases")
    path("cases", create, name="cases"),
    path('upload/', upload_file, name="upload_file"),
    path('chat_constituition/', chat_constituition, name="chat_constituition"),
    # path('case_search/', case_search, name="case_search"),
    path('case_search/', text_index_search_case, name="case_search"),
]
