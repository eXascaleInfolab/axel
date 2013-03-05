"""Url mappings"""
from django.conf.urls import patterns, url
from django.contrib.auth.decorators import user_passes_test
from axel.stats.views import *


urlpatterns = patterns('axel.stats.views',
    url(r'^$', CollocationMainView.as_view(), name='stats'),

    # Distributions
    url(r'^(?P<model_name>[^/]+)/$', CollocationStats.as_view(), name='count_dist'),
    url(r'^(?P<model_name>[^/]+)/ci/$', ConceptIndexStats.as_view(), name='ci_stats'),
    url(r'^(?P<model_name>[^/]+)/ngram_participation/$',
        NgramParticipationView.as_view(), name='ngram_participation_stats'),
    url(r'^(?P<model_name>[^/]+)/pos_dist/$',  NgramPOSView.as_view(), name='pos_dist'),
    url(r'^(?P<model_name>[^/]+)/ngram_measures/$',  NgramMeasureScoringView.as_view(),
        name='ngram_measures'),

    # Filtering
    url(r'^(?P<model_name>[^/]+)/filter/$', FilteredCollectionModelView.as_view(),
        name='colloc_filter'),

    # Clear cache
    url(r'^(?P<model_name>[^/]+)/clear/$', user_passes_test(lambda u: u.is_superuser)(
        ClearCachedAttrView.as_view()), name='clear_attribute')
)
