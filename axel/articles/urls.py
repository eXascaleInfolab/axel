"""Url mappings"""
from django.conf.urls import patterns, url

from axel.articles.views import PDFCollocationsView

urlpatterns = patterns('',
    url(r'pdfcollocations/$', PDFCollocationsView.as_view(), name='pdf_collocations'),
)
