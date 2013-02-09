from collections import defaultdict
import json
import numpy

from django.core.cache import cache
from django.views.generic import TemplateView
from test_collection.views import CollectionModelView, _get_model_from_string, TestCollectionOverview

from axel.articles.utils.concepts_index import WORDS_SET, CONCEPT_PREFIX
from axel.articles.utils.nlp import build_ngram_index


class CollocationMainView(TestCollectionOverview):
    """Main conceptual search view"""
    template_name = "stats/overview.html"


class CollocationStats(TemplateView):
    """Stats about raw counts distribution"""
    template_name = "stats/collocations.html"

    def get_context_data(self, **kwargs):
        """Add form to context"""
        context = super(CollocationStats, self).get_context_data(**kwargs)
        model = _get_model_from_string(self.kwargs['model_name'])
        #counts, bins = numpy.histogram(model.objects.values_list('count', flat=True), bins=10)
        #counts = [x+1 for x in counts]
        counts = defaultdict(lambda: 0)
        for count in model.objects.values_list('count', flat=True):
            counts[count]+=1
        context['histogram_data'] = str(counts.items()).replace('(', '[').replace(')', ']')
        context['collocations'] = model.objects.order_by('-count').values_list('count',
            'ngram')[:10]
        return context


class ConceptIndexStats(TemplateView):
    """Displays statistics about concept index:
    N-gram word distribution
    """

    template_name = "stats/conceptindex.html"
    def get_context_data(self, **kwargs):
        """Add data to context"""
        context = super(ConceptIndexStats, self).get_context_data(**kwargs)

        model = _get_model_from_string(self.kwargs['model_name'])
        global_word_set = cache.get(WORDS_SET)
        counts = defaultdict(lambda: 0)
        for word in global_word_set:
            counts[len(cache.get(CONCEPT_PREFIX+word))] += 1

        context['histogram_data'] = str(counts.items()).replace('(', '[').replace(')', ']')
        context['word_count'] = len(global_word_set)
        context['concept_count'] = model.objects.count()

        word_counts = defaultdict(lambda: 0)
        for collocation in model.objects.values_list("ngram", flat=True):
            word_counts[len(collocation.split())] += 1
        context['col_word_len_hist'] = str(word_counts.items()).replace('(', '[').replace(')', ']')
        return context


class FilteredCollectionModelView(CollectionModelView):
    """Filtered collection view, for searching using query"""
    query = None

    def get(self, request, *args, **kwargs):
        """get form args"""
        self.query = request.GET.get('query', '')
        return super(FilteredCollectionModelView, self).get(request, *args, **kwargs)

    def generate_queryset(self, model):
        """filter ngram by query here, filter only unjudged results"""
        return self.total_queryset.filter(ngram__icontains=self.query)


class NgramParticipationView(TemplateView):
    """View to draw ngram participation graph, d3.js"""
    template_name='stats/graph_vis/ngram_particiation.html'

    def get_context_data(self, **kwargs):
        """Add nodes and links to the context"""
        context = super(NgramParticipationView, self).get_context_data(**kwargs)
        model = _get_model_from_string(self.kwargs['model_name'])
        # nodes are simply ngrams
        links = []
        all_ngrams = list(model.objects.filter(tags__is_relevant__isnull=False).values_list(
            'ngram', flat=True))
        rel_ngrams = set(model.objects.filter(tags__is_relevant=True).values_list(
            'ngram', flat=True))
        # Sort from longest to shortest, we use this in computing connections
        all_ngrams.sort(key=lambda x: len(x)+len(x.split()), reverse=True)

        ngrams_set = set(all_ngrams)
        participation_dict = defaultdict(list)
        for ngram in all_ngrams:
            if ngram in participation_dict:
                for ngram_1 in participation_dict[ngram]:
                    links.append((ngram, ngram_1))
                # replace with current ngram
                for ngram_i in ngrams_set.intersection(build_ngram_index(ngram).keys()):
                    participation_dict[ngram_i] = [ngram]
            else:
                # append current ngram
                for ngram_i in ngrams_set.intersection(build_ngram_index(ngram).keys()):
                    participation_dict[ngram_i].append(ngram)

        # keep only connected components
        connected_nodes = list(set(zip(*links)[0]).union(set(zip(*links)[1])))
        node_dict = dict([(node,i) for i, node in enumerate(connected_nodes)])


        links = [{'source':node_dict[source], 'target': node_dict[target]} for source,
                                                                            target in links]
        nodes = [{"name": ngram, "rel": ngram in rel_ngrams} for ngram in connected_nodes]
        context['data'] = json.dumps({'nodes': nodes, 'links': links})
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

