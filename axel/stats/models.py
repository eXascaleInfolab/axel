from __future__ import division
import json

from collections import defaultdict
from django.db import models
from django import forms
from django.db.models.signals import post_save
from django.dispatch import receiver

from axel.articles.utils.db import db_cache_simple, db_cache
from axel.libs.external_match import perform_match
import axel.stats.scores as scores


class Collocation(models.Model):
    """Aggregated collocation statistics model"""
    ngram = models.CharField(max_length=255)
    count = models.IntegerField(default=1)
    # extra fields will store pre-computed scores
    _extra_fields = models.TextField(default='{}')
    _df_score = models.IntegerField(null=True, blank=True)
    _max_pos_tag = models.CharField(null=True, max_length=100)
    _pos_tag_prev = models.CharField(null=True, max_length=100)
    _pos_tag_after = models.CharField(null=True, max_length=100)
    _ms_ngram_score = models.DecimalField(default=0, decimal_places=6, max_digits=9)

    CLUSTER_ID = 'ABSTRACT'
    CACHED_FIELDS = ()
    FILTERED_FIELDS = ()

    class Meta:
        """Meta info"""
        abstract = True
        ordering = ['-count']

    def __unicode__(self):
        """String representation"""
        return u'{0}'.format(self.ngram)

    @property
    def _articlecollocations(self):
        """
        :rtype: QuerySet
        """
        from axel.articles.models import ArticleCollocation
        return ArticleCollocation.objects.filter(article__cluster_id=self.CLUSTER_ID)

    @property
    def extra_fields(self):
        """Load from json"""
        return json.loads(self._extra_fields)

    @extra_fields.setter
    def extra_fields(self, value):
        """Convert to json"""
        self._extra_fields = json.dumps(value)

    @property
    @db_cache('extra_fields')
    def source(self):
        """
        Return source of the field, empty list by default
        :rtype: list
        """
        return []

    @property
    def wikipedia_text(self):
        import requests
        from django.utils.html import strip_tags
        from axel.libs import nlp
        if not 'dbpedia' in self.source:
            return ''
        query = u'http://en.wikipedia.org/w/api.php?action=parse&page={0}&redirects&format=json'
        result = json.loads(requests.get(query.format(self.ngram.replace(' ', '_'))).text)
        try:
            return nlp.Stemmer.stem_wordnet(strip_tags(result['parse']['text']['*']))
        except KeyError:
            return ''

    @property
    def count_score(self):
        """Proxy to the count, need for test collection app to pick up the _score"""
        return self.count

    @property
    @db_cache_simple
    def max_pos_tag(self):
        """
        Maximal pos tag for the collection
        :rtype: string
        """
        from axel.articles.models import CLUSTERS_DICT
        pos_tags = defaultdict(lambda: 0)
        for ngram in CLUSTERS_DICT[self.CLUSTER_ID].objects.filter(ngram=self.ngram):
            for pos_tag, count in ngram.pos_tag:
                pos_tags[' '.join(pos_tag)] += count
        return max(pos_tags.items(), key=lambda x: x[1])[0]

    @property
    @db_cache_simple
    def pos_tag_prev(self):
        from axel.articles.models import CLUSTERS_DICT
        tags = defaultdict(lambda: 0)
        for ngram in CLUSTERS_DICT[self.CLUSTER_ID].objects.filter(ngram=self.ngram):
            for tag, count in ngram.pos_tag_prev:
                tags[tag] += count
        return tags.items()

    @property
    @db_cache_simple
    def pos_tag_after(self):
        from axel.articles.models import CLUSTERS_DICT
        tags = defaultdict(lambda: 0)
        for ngram in CLUSTERS_DICT[self.CLUSTER_ID].objects.filter(ngram=self.ngram):
            for tag, count in ngram.pos_tag_after:
                tags[tag] += count
        return tags.items()

    @property
    @db_cache_simple
    def df_score(self):
        """
        Document frequency
        :rtype: int
        """
        from axel.articles.models import ArticleCollocation
        return ArticleCollocation.objects.filter(ngram=self.ngram).count()

    @property
    def occur_distribution(self):
        """
        :rtype: str
        :returns: histogram data in a string form suitable for highcharts
        """
        counts = defaultdict(lambda: 0)
        for count in self._articlecollocations.filter(ngram=self.ngram).values_list('count',
                                                                                    flat=True):
            counts[count] += 1

        histogram_data = str(sorted(counts.items())).replace('(', '[').replace(')', ']')
        return histogram_data


class Collocations(Collocation):
    """Aggregated collocation statistics model for Computer Science"""
    CACHED_FIELDS = ('context',)
    FILTERED_FIELDS = (('_pos_tag', 'Part of Speech', forms.CharField),)
    CLUSTER_ID = 'CS_COLLOCS'


class SWCollocations(Collocation):
    """
    collocation for ScienceWISE
    everything is the same except table name
    """
    CLUSTER_ID = 'SW_COLLOCS'
    CACHED_FIELDS = ('context',)

    @property
    def is_ontological(self):
        """
        True if concept appears in ontology, False otherwise.
        """
        return self.ngram in scores.ontology


def set_source_field(sender, instance, created, **kwargs):
    """
    Increment collocation count on create for ArticleCollocation
    :type instance: ArticleCollocation
    """
    if kwargs.get('raw'):
        return
    if created:
        instance.extra_fields = {'source': perform_match(instance)}
        instance.save_base(raw=True)

post_save.connect(set_source_field, sender=Collocations)
post_save.connect(set_source_field, sender=SWCollocations)

STATS_CLUSTERS_DICT = dict([(model.CLUSTER_ID, model) for model in (Collocations, SWCollocations)])