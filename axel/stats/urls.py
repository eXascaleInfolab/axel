"""Url mappings"""
from django.conf.urls import patterns, url
from axel.stats.views import CollocationStats


urlpatterns = patterns('axel.stats.views',
    url(r'^$', CollocationStats.as_view(), name='stats'),
)
