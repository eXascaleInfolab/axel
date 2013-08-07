from __future__ import division
import json

from collections import defaultdict
from django.db import models
from django.db.models import Q, Sum
from django import forms

import operator
from axel.articles.utils.db import db_cache_simple, db_cache
import axel.stats.scores as scores
from axel.libs.utils import get_contexts_ngrams


class Collocation(models.Model):
    """Aggregated collocation statistics model"""
    ngram = models.CharField(max_length=255)
    count = models.IntegerField(default=1)
    # extra fields will store pre-computed scores
    _extra_fields = models.TextField(default='{}')
    _pos_tag = models.CharField(max_length=255, null=True, blank=True)
    _df_score = models.IntegerField(null=True, blank=True)
    _is_wiki = models.BooleanField(default=False)
    _ms_ngram_score = models.DecimalField(default=0, decimal_places=6, max_digits=9)

    # required for TestCollection upload
    SYNC_FIELD = 'ngram'
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
            return nlp.Stemmer.stem_wordnet(strip_tags(result['parse']['text']))
        except KeyError:
            return ''

    @property
    def count_score(self):
        """Proxy to the count, need for test collection app to pick up the _score"""
        return self.count

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
    @db_cache_simple
    def pos_tag(self):
        """
        Defines part-of-speech tag for ngram
        :return: Part-of-Speech tag
        :rtype: unicode
        """
        return scores.pos_tag(self.ngram, self.all_contexts(func=get_contexts_ngrams))


    @property
    @db_cache('extra_fields')
    def pos_tag_prev(self):
        """
        Retrieves part-of-speech tag for the word before ngram
        :return: dict of Part-of-Speech tags with scores
        :rtype: dict
        """
        return scores.pos_tag_pos(self.ngram, self.all_contexts(func=get_contexts_ngrams))

    @property
    @db_cache('extra_fields')
    def pos_tag_after(self):
        """
        Retrieves part-of-speech tag for the word before ngram
        :return: dict of Part-of-Speech tags with scores
        :rtype: dict
        """
        return scores.pos_tag_pos(self.ngram, self.all_contexts(func=get_contexts_ngrams), tag_pos=1)

    @property
    def often_word_local(self):
        """How many times do words from the ngram occur in other ngrams from the same article"""
        argument_list = []
        for word in self.ngram.split():
            argument_list.append(Q(**{'ngram__regex': r'\b'+word+r'\b'}))
        query = reduce(operator.or_, argument_list)

        article_ids = self._articlecollocations.filter(ngram=self.ngram).values_list(
            'article', flat=True)
        score = self._articlecollocations.filter(article__id__in=article_ids).filter(query)\
                                            .count() - len(article_ids)
        return score

    @property
    @db_cache('extra_fields')
    def often_consumed_score(self):
        """How often does an ngram gets consumed by a bigger one"""
        score = self.__class__.objects.filter(ngram__contains=self.ngram).aggregate(
            count=Sum('count'))['count']
        return score // self.count - 1

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
    CACHED_FIELDS = ('context', 'acm_score')
    FILTERED_FIELDS = (('_pos_tag', 'Part of Speech', forms.CharField),)


class SWCollocations(Collocation):
    """
    collocation for ScienceWISE
    everything is the same except table name
    """
    CACHED_FIELDS = ('context', 'partial_word_score', 'partial_ngram_score',
                     'partial_ont_score')

    @property
    @db_cache('extra_fields')
    def is_ontological(self):
        """
        True if concept appears in ontology, False otherwise.
        """
        return self.ngram in scores.ontology

    @property
    @db_cache('extra_fields')
    def partial_word_score(self):
        """
        Sum of the counts of words from a given collocation in the ontology
        (how often a word appears as a part of a concept in the ontology).
        """
        return scores.get_word_concept_score(self.ngram)

    @property
    @db_cache('extra_fields')
    def partial_ngram_score(self):
        """
        Sum of the counts of FULL NGRAM in the ontology
        (How often a full ngram appears as a part of a concept in the ontology)
        """
        return scores.get_ngram_concept_score(self.ngram)

    @property
    @db_cache('extra_fields')
    def partial_ont_score(self):
        """How often does any concept from the ontology occur in the NGRAM"""
        return scores.get_concept_ngram_score(self.ngram)
