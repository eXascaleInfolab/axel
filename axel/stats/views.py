from collections import defaultdict
import numpy
from haystack.query import SearchQuerySet

from django.core.cache import cache
from django.core.urlresolvers import reverse
from django.http import HttpResponseRedirect
from django.views.decorators.http import require_POST
from django.views.generic import TemplateView

from axel.articles.models import ArticleCollocation, Article
from axel.articles.utils.concepts_index import WORDS_SET, CONCEPT_PREFIX
from axel.stats.models import Collocation


class CollocationStats(TemplateView):
    """Main conceptual search view"""

    template_name = "stats/collocations.html"

    def get_context_data(self, **kwargs):
        """Add form to context"""
        context = super(CollocationStats, self).get_context_data(**kwargs)
        counts, bins = numpy.histogram(Collocation.objects.values_list('count', flat=True),
            bins=10)
        counts = [x+1 for x in counts]
        context['histogram_data'] = str(zip(bins, counts)).replace('(', '[').replace(')', ']')
        context['collocations'] = Collocation.objects.order_by('-count').values_list('count',
            'ngram')[:10]
        return context


class ConceptIndexStats(TemplateView):
    """displays statistics about concept index"""

    template_name = "stats/conceptindex.html"
    def get_context_data(self, **kwargs):
        """Add data to context"""
        context = super(ConceptIndexStats, self).get_context_data(**kwargs)
        global_word_set = cache.get(WORDS_SET)
        counts = defaultdict(lambda: 0)
        for word in global_word_set:
            counts[len(cache.get(CONCEPT_PREFIX+word))] += 1

        context['histogram_data'] = str(counts.items()).replace('(', '[').replace(')', ']')
        context['word_count'] = len(global_word_set)
        context['concept_count'] = Collocation.objects.count()

        word_counts = defaultdict(lambda: 0)
        for collocation in Collocation.objects.values_list("ngram", flat=True):
            word_counts[len(collocation.split())] += 1
        context['col_word_len_hist'] = str(word_counts.items()).replace('(', '[').replace(')', ']')
        return context



#@require_POST
#def collocations_update(request):
#    """Update collocations by searching existing ones"""
#    for collocation in Collocations.objects.all():
#        cur_docs = SearchQuerySet().filter(content__exact=collocation.ngram).values_list('id',
#            flat=True)
#        # strip model
#        cur_docs = [doc.split('.')[-1] for doc in cur_docs]
#        if len(cur_docs) > collocation.count:
#            docs = set(Article.objects.exclude(articlecollocation__ngram=collocation.ngram).
#                    filter(id__in=cur_docs).values_list('id', 'stemmed_text'))
#            for doc_id, text in docs:
#                c_count = text.count(collocation.ngram)
#                ArticleCollocation.objects.create(article_id=doc_id,
#                    ngram=collocation.ngram, count=c_count)
#
#    return HttpResponseRedirect(reverse('stats'))

