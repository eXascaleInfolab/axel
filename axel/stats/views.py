from collections import defaultdict
import json
from django.contrib.contenttypes.models import ContentType

from django.core.cache import cache
from django.core.urlresolvers import reverse
from django.http import HttpResponseRedirect
from django.views.generic import TemplateView, FormView
from test_collection.models import TaggedCollection
from test_collection.views import CollectionModelView, _get_model_from_string, TestCollectionOverview

from axel.articles.utils.concepts_index import WORDS_SET, CONCEPT_PREFIX
from axel.articles.utils.nlp import build_ngram_index
from axel.stats.forms import ScoreCacheResetForm


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


class NgramPOSView(TemplateView):
    """View to draw ngram participation graph, d3.js"""
    template_name='stats/graph_vis/pos_distribution.html'

    def get_context_data(self, **kwargs):
        """Add nodes and links to the context"""
        context = super(NgramPOSView, self).get_context_data(**kwargs)
        model = _get_model_from_string(self.kwargs['model_name'])
        ct = ContentType.objects.get_for_model(model)

        relevant_ids = set(TaggedCollection.objects.filter(content_type=ct,
            is_relevant=True).values_list('object_id', flat=True))
        irrelevant_ids = set(TaggedCollection.objects.filter(content_type=ct,
            is_relevant=False).values_list('object_id', flat=True))

        all_tags = set()

        context['correct_data'] = defaultdict(lambda:0)
        context['incorrect_data'] = defaultdict(lambda:0)
        context['unjudged_data'] = defaultdict(lambda:0)
        for obj in model.objects.all():
            tag = str(obj.pos_tag)
            all_tags.add(tag)
            if obj.id in relevant_ids:
                context['correct_data'][tag] += 1
            elif obj.id in irrelevant_ids:
                context['incorrect_data'][tag] += 1
            else:
                context['unjudged_data'][tag] += 1

        context['correct_data'] = [context['correct_data'][tag] for tag in all_tags]
        context['incorrect_data'] = [context['incorrect_data'][tag] for tag in all_tags]
        context['unjudged_data'] = [context['unjudged_data'][tag] for tag in all_tags]

        context['categories'] = sorted(all_tags)
        return context


class ClearCachedAttrView(FormView):
    """Extract and display collocations from pdf document"""
    form_class = ScoreCacheResetForm
    template_name = "test_collection/partial/clear_cache_form.html"

    def get_form(self, form_class):
        """
        Returns an instance of the form to be used in this view.
        """
        self.model_name = self.kwargs['model_name']
        model = _get_model_from_string(self.model_name)
        return form_class(model, **self.get_form_kwargs())

    def form_valid(self, form):
        """
        clear cache
        """
        attribute = form.cleaned_data['attr']
        model = _get_model_from_string(self.model_name)
        for obj in model.objects.all():
            fields = obj.extra_fields
            if attribute in fields:
                del fields[attribute]
                obj.extra_fields = fields
                obj.save()

        next = self.request.GET.get('next', reverse('testcollection_model', args=[self.model_name]))
        return HttpResponseRedirect(next)

    def get_context_data(self, **kwargs):
        """Add nodes and links to the context"""
        context = super(ClearCachedAttrView, self).get_context_data(**kwargs)
        context['model_name'] = self.model_name
        return context

