from __future__ import division

from collections import defaultdict
from django.db import models
from django.db.models import Q, Sum
import operator
from axel.articles.models import ArticleCollocation
import axel.articles.utils.sw_indexes as sw
from axel.libs.utils import get_contexts


class Collocation(models.Model):
    """Aggregated collocation statistics model"""
    ngram = models.CharField(max_length=255)
    count = models.IntegerField(default=1)

    class Meta:
        """Meta info"""
        abstract = True
        ordering = ['-count']

    def __unicode__(self):
        """String representation"""
        return u'{0}'.format(self.ngram)

    @property
    def context(self):
        """Get random context for collocation, used in collocation list view"""
        article =  ArticleCollocation.objects.filter(ngram=self.ngram)[0].article
        # prevent contexts from bigger ngrams
        bigger_ngrams = ArticleCollocation.objects.filter(article=article,
            ngram__contains=self.ngram).exclude(ngram=self.ngram).values_list('ngram', flat=True)
        return get_contexts(article.stemmed_text, self.ngram, bigger_ngrams).next()

    @property
    def all_contexts(self):
        """Get all contexts for detailed view page"""
        contexts = []
        for text, article_id in  ArticleCollocation.objects.filter(ngram=self.ngram).values_list(
            'article__stemmed_text', 'article'):
            bigger_ngrams = ArticleCollocation.objects.filter(article__id=article_id,
                ngram__contains=self.ngram).exclude(ngram=self.ngram).values_list('ngram', flat=True)
            contexts.extend([context for context in get_contexts(text, self.ngram, bigger_ngrams)])
        return contexts

    @property
    def count_score(self):
        """Proxy to the count, need for test collection app to pick up the _score"""
        return self.count

    @property
    def partial_match_score(self):
        """
        Sum of the counts of words from a given collocation in the ontology
        (how often a word appears as a part of a concept in the ontology).
        """
        return sw.get_concept_score(self.ngram)

    @property
    def often_score_glob(self):
        """How many articles do contain an ngram"""
        return ArticleCollocation.objects.filter(ngram=self.ngram).count()

    @property
    def often_word_local(self):
        """How many times do words from the ngram occur in other ngrams from the same article"""
        argument_list = []
        for word in self.ngram.split():
            argument_list.append(Q(**{'ngram__regex': r'\b'+word+r'\b'}))
        query = reduce(operator.or_, argument_list)

        article_ids = ArticleCollocation.objects.filter(ngram=self.ngram).values_list(
            'article', flat=True)
        score = ArticleCollocation.objects.filter(article__id__in=article_ids).filter(query)\
                                            .count() - len(article_ids)
        return score

    @property
    def often_consumed_score(self):
        """How often does an ngram gets consumed by a bigger one"""
        score = Collocation.objects.filter(ngram__contains=self.ngram).aggregate(
            count=Sum('count'))['count']
        return score // self.count - 1

    @property
    def occur_distribution(self):
        """
        :rtype: str
        :returns: histogram data in a string form suitable for highcharts
        """
        counts = defaultdict(lambda: 0)
        for count in ArticleCollocation.objects.filter(ngram=self.ngram).values_list('count',
                                flat=True):
            counts[count] += 1

        histogram_data = str(counts.items()).replace('(', '[').replace(')', ']')
        return histogram_data


class Collocations(Collocation):
    """Aggregated collocation statistics model for Computer Science"""


class SWCollocations(Collocation):
    """
    collocation for ScienceWISE
    everything is the same except table name
    """

