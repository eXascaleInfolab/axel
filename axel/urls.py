from django.conf.urls import patterns, include, url

from django.contrib import admin
from axel.articles.views import ConceptualSearchView

admin.autodiscover()

urlpatterns = patterns('',
    # Examples:
    # url(r'^axel/', include('axel.foo.urls')),

    # Uncomment the admin/doc line below to enable admin documentation:
    # url(r'^admin/doc/', include('django.contrib.admindocs.urls')),

    url(r'^admin/', include(admin.site.urls)),
    (r'^search/', include('haystack.urls')),
    (r'^articles/', include('axel.articles.urls')),
    (r'^stats/', include('axel.stats.urls')),
    (r'^testc/', include('test_collection.urls')),
    url(r'^$', ConceptualSearchView.as_view(), name='main')
)
