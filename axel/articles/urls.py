"""Url mappings"""
from django.conf.urls import patterns, url

from axel.articles.views import PDFCollocationsView, ArticleList


urlpatterns = patterns('',
    url(r'pdfcollocations/$', PDFCollocationsView.as_view(), name='pdf_collocations'),
    url(r'^$', ArticleList.as_view()),
)
