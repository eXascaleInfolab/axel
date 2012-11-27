"""Url mappings"""
from django.conf.urls import patterns, url
from axel.stats.views import CollocationStats, ConceptIndexStats


urlpatterns = patterns('axel.stats.views',
    url(r'^$', CollocationStats.as_view(), name='stats'),
    url(r'^ci/$', ConceptIndexStats.as_view(), name='ci_stats'),
)
