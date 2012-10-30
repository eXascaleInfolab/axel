"""Url mappings"""
from django.conf.urls import patterns, url

from axel.articles.views import PDFCollocationsView, ArticleList, ArticleDetailView,\
    ConceptAutocompleteView


urlpatterns = patterns('',
    url(r'pdfcollocations/$', PDFCollocationsView.as_view(), name='pdf_collocations'),
    url(r'^$', ArticleList.as_view(), name='articles'),
    url(r'^(?P<pk>\d+)/$', ArticleDetailView.as_view(), name='article_detail'),
    url(r'^concept_autocomplete/$', ConceptAutocompleteView.as_view(), name='concept_autocomplete')
)
