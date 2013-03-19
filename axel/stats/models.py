from __future__ import division
import json

from collections import defaultdict
from django.contrib.contenttypes import generic
from django.db import models
from django.db.models import Q, Sum
from django import forms

import operator
from test_collection.models import TaggedCollection
from axel.articles.models import ArticleCollocation
import axel.stats.scores as scores
from axel.libs.utils import get_contexts, get_contexts_ngrams


# TODO: make a blog post out of a technique
class db_cache(object):
    """
    Decorator to cache the expensively computed field in some other field
    that supports dict-like assignment
    """

    def __init__(self, model_field):
        """
        :param model_field: field of the model to store and retrieve field from
        """
        self.model_field = model_field

    def __call__(self, f):
        def wrapper(object):
            fields = getattr(object, self.model_field)
            if f.__name__ in fields:
                return fields[f.__name__]
            else:
                value = f(object)
                fields[f.__name__] = value
                setattr(object, self.model_field, fields)
                object.save()
                return value
        return wrapper


def db_cache_simple(func):
    def wrapper(self):
        value = getattr(self, '_' + func.__name__)
        if value:
            return value
        else:
            value = func(self)
            setattr(self, '_' + func.__name__, value)
            self.save_base(raw=True)
            return value
    return wrapper


class Collocation(models.Model):
    """Aggregated collocation statistics model"""
    ngram = models.CharField(max_length=255)
    count = models.IntegerField(default=1)
    tags = generic.GenericRelation(TaggedCollection)
    # extra fields will store pre-computed scores
    _extra_fields = models.TextField(default='{}')
    _pos_tag = models.CharField(max_length=255, null=True, blank=True)
    _df_score = models.IntegerField(null=True, blank=True)

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
    def context(self):
        """
        Get random context for collocation, used in collocation list view,
        :rtype: unicode
        :returns: context if found, ngram itself otherwise
        """
        article = self._articlecollocations.filter(ngram=self.ngram)[0].article
        # prevent contexts from bigger ngrams
        bigger_ngrams = self._articlecollocations.filter(article=article,
            ngram__contains=self.ngram).exclude(ngram=self.ngram).values_list('ngram', flat=True)
        context = next(get_contexts(article.text, self.ngram, bigger_ngrams),
           self.ngram)
        return context

    @property
    @db_cache('extra_fields')
    def source(self):
        """Return source of the field, None by default"""
        return None

    def all_contexts(self, func=get_contexts):
        """
        Get all contexts for detailed view page
        :rtype: list
        :returns: contexts if found, [ngram] otherwise
        """
        contexts = []
        for text, article_id in self._articlecollocations.filter(ngram=self.ngram).values_list(
            'article__text', 'article'):
            bigger_ngrams = self._articlecollocations.filter(article__id=article_id,
                ngram__contains=self.ngram).exclude(ngram=self.ngram).values_list('ngram', flat=True)
            for context in func(text, self.ngram, bigger_ngrams):
                contexts.append(context)
        return contexts

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
        return scores.pos_tag_prev(self.ngram, self.all_contexts(func=get_contexts_ngrams))

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
    CLUSTER_ID = 'CS_COLLOCS'
    CACHED_FIELDS = ('context', 'acm_score')
    FILTERED_FIELDS = (('_pos_tag', 'Part of Speech', forms.CharField),)

    @property
    @db_cache('extra_fields')
    def acm_score(self):
        """
        Get ACM search score
        """
        return 0
        #return scores.acm_search_result_count(self.ngram)


class SWCollocations(Collocation):
    """
    collocation for ScienceWISE
    everything is the same except table name
    """
    CLUSTER_ID = 'SW_COLLOCS'
    CACHED_FIELDS = ('context', 'partial_word_score', 'partial_ngram_score',
                     'partial_ont_score')

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


CLUSTERS_DICT = dict([(model.CLUSTER_ID, model) for model in (Collocations, SWCollocations)])

