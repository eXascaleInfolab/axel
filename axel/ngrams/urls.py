from django.conf.urls import patterns, url

from axel.ngrams.views import SentenceList, SentenceDetailView


urlpatterns = patterns('axel.ngrams.views',
    url(r'^$', SentenceList.as_view(), name='sentences'),
    url(r'^(?P<pk>\d+)/$', SentenceDetailView.as_view(), name='sentence_detail'),
)
