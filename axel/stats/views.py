from __future__ import division
from collections import defaultdict, OrderedDict, Counter
import json
import re
from numpy import array

from django.contrib.contenttypes.models import ContentType
from django.core.cache import cache
from django.core.urlresolvers import reverse
from django.http import HttpResponseRedirect
from django.views.generic import TemplateView, FormView
from test_collection.models import TaggedCollection
from test_collection.views import CollectionModelView, _get_model_from_string,\
    TestCollectionOverview
from axel.articles.models import ArticleCollocation, Article

from axel.articles.utils.concepts_index import WORDS_SET, CONCEPT_PREFIX
from axel.articles.utils.nlp import build_ngram_index
from axel.libs.mixins import AttributeFilterView
from axel.stats import scores
from axel.stats.forms import ScoreCacheResetForm, NgramBindingForm
from axel.stats.scores.ngram_ranking import NgramMeasureScoring


class CollocationMainView(TestCollectionOverview):
    """Main conceptual search view"""
    template_name = "stats/overview.html"


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


class CollocationAttributeFilterView(AttributeFilterView):
    """Class to define all defaults for Collocation objects"""
    model_fields_attr = 'FILTERED_FIELDS'


class ConceptIndexStats(TemplateView):
    # TODO: inherit from FILTERVIEW?
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
            counts[len(cache.get(CONCEPT_PREFIX + word))] += 1

        context['histogram_data'] = str(counts.items()).replace('(', '[').replace(')', ']')
        context['word_count'] = len(global_word_set)
        context['concept_count'] = model.objects.count()

        word_counts = defaultdict(lambda: 0)
        for collocation in model.objects.values_list("ngram", flat=True):
            word_counts[len(collocation.split())] += 1
        context['col_word_len_hist'] = str(word_counts.items()).replace('(', '[').replace(')', ']')
        return context


class NgramParticipationView(CollocationAttributeFilterView):
    """View to draw ngram participation graph, d3.js"""
    template_name = 'stats/graph_vis/ngram_particiation.html'

    def get_context_data(self, **kwargs):
        """Add nodes and links to the context"""
        context = super(NgramParticipationView, self).get_context_data(**kwargs)
        # nodes are simply ngrams
        links = []
        irrel_ngrams = set(self.queryset.filter(tags__is_relevant=False).values_list(
            'ngram', flat=True))
        rel_ngrams = set(self.queryset.filter(tags__is_relevant=True).values_list(
            'ngram', flat=True))

        all_ngrams = list(self.queryset.values_list('ngram', flat=True))
        # Sort from longest to shortest, we use this in computing connections
        all_ngrams.sort(key=lambda x: len(x) + len(x.split()), reverse=True)

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
        node_dict = dict([(node, i) for i, node in enumerate(connected_nodes)])

        links = [{'source':node_dict[source], 'target': node_dict[target]}
                 for source, target in links]

        def _get_rel_info(ngram):
            if ngram in rel_ngrams:
                return 1
            elif ngram in irrel_ngrams:
                return -1
            return 0

        nodes = [{"name": ngram, "rel": _get_rel_info(ngram)} for ngram in
                 connected_nodes]
        context['data'] = json.dumps({'nodes': nodes, 'links': links})
        return context


class NgramPOSView(TemplateView):
    """View to draw ngram participation graph, d3.js"""
    template_name = 'stats/graph_vis/pos_distribution.html'

    def _parse_rules(self):
        """
        :returns: parsed rules dict to compress POS tags
        :rtype: dict
        """
        rules_dict = OrderedDict()
        # add three extra forms to extend initial forms
        names = self.request.GET.getlist('groupname', [''])
        regexes = self.request.GET.getlist('regex', [''])
        if names[-1] != '':
            names.append('')
        if regexes[-1] != '':
            regexes.append('')
        for name, regex in zip(names, regexes):
            if name and regex:
                rules_dict[name] = re.compile(regex)
        self.regex_groups = zip(names, regexes)
        return rules_dict

    def get_context_data(self, **kwargs):
        """Add nodes and links to the context"""
        context = super(NgramPOSView, self).get_context_data(**kwargs)
        model = _get_model_from_string(self.kwargs['model_name'])
        ct = ContentType.objects.get_for_model(model)

        rules_dict = self._parse_rules()

        relevant_ids = set(TaggedCollection.objects.filter(content_type=ct,
                           is_relevant=True).values_list('object_id', flat=True))
        irrelevant_ids = set(TaggedCollection.objects.filter(content_type=ct,
                             is_relevant=False).values_list('object_id', flat=True))

        all_tags = set()

        correct_data = defaultdict(lambda: 0)
        incorrect_data = defaultdict(lambda: 0)
        unjudged_data = defaultdict(lambda: 0)
        for obj in model.objects.all():
            tag = str(scores.compress_pos_tag(obj.pos_tag, rules_dict))
            all_tags.add(tag)
            if obj.id in relevant_ids:
                correct_data[tag] += 1
            elif obj.id in irrelevant_ids:
                incorrect_data[tag] += 1
            else:
                unjudged_data[tag] += 1

        all_tags = sorted(all_tags)
        context['categories'] = all_tags
        context['top_relevant_data'] = [(tag, correct_data[tag], incorrect_data[tag],
                                         unjudged_data[tag]) for tag in all_tags
                if int(correct_data[tag] / (incorrect_data[tag] + 0.1)) > 10]

        context['top_irrelevant_data'] = [(tag, correct_data[tag], incorrect_data[tag],
                                           unjudged_data[tag]) for tag in all_tags
                if int(incorrect_data[tag] / (correct_data[tag] + 0.1)) > 10]

        context['correct_data'] = [correct_data[tag] for tag in all_tags]
        context['incorrect_data'] = [incorrect_data[tag] for tag in all_tags]
        context['unjudged_data'] = [unjudged_data[tag] for tag in all_tags]

        # Add regex groups to populate forms
        context['regex_groups'] = self.regex_groups

        return context


class NgramPrevPOSView(CollocationAttributeFilterView):
    template_name = 'stats/graph_vis/pos_distribution.html'

    def get_context_data(self, **kwargs):
        """Add nodes and links to the context"""
        context = super(NgramPrevPOSView, self).get_context_data(**kwargs)

        irrel_ids = set(self.queryset.filter(tags__is_relevant=False).values_list('id', flat=True))
        rel_ids = set(self.queryset.filter(tags__is_relevant=True).values_list('id', flat=True))

        all_tags = set()

        correct_data = defaultdict(lambda: 0)
        incorrect_data = defaultdict(lambda: 0)
        unjudged_data = defaultdict(lambda: 0)
        for obj in self.queryset:
            for tag, count in obj.pos_tag_prev.iteritems():
                if count > 10:
                    tag = str(tag)
                    all_tags.add(tag)
                    if obj.id in rel_ids:
                        correct_data[tag] += 1
                    elif obj.id in irrel_ids:
                        incorrect_data[tag] += 1
                    else:
                        unjudged_data[tag] += 1

        all_tags = sorted(all_tags)
        context['categories'] = all_tags
        context['top_relevant_data'] = [(tag, correct_data[tag], incorrect_data[tag],
                                         unjudged_data[tag]) for tag in all_tags
                                        if int(correct_data[tag] / (incorrect_data[tag] + 0.1)) > 10]

        context['top_irrelevant_data'] = [(tag, correct_data[tag], incorrect_data[tag],
                                           unjudged_data[tag]) for tag in all_tags
                                          if int(incorrect_data[tag] / (correct_data[tag] + 0.1)) > 10]

        context['correct_data'] = [correct_data[tag] for tag in all_tags]
        context['incorrect_data'] = [incorrect_data[tag] for tag in all_tags]
        context['unjudged_data'] = [unjudged_data[tag] for tag in all_tags]

        return context


class NgramMeasureScoringView(CollocationAttributeFilterView):
    """View to get """
    template_name = 'stats/graph_vis/ngram_scoring_distribution.html'

    def get_context_data(self, **kwargs):
        """Add nodes and links to the context"""
        context = super(NgramMeasureScoringView, self).get_context_data(**kwargs)
        scores = NgramMeasureScoring.get_scores(self.queryset)
        context['graph_data'] = [(measure_name, str(data).replace('(', '[').replace(')', ']'))
                                 for measure_name, data in scores.iteritems()]
        return context


class NgramWordBindingDistributionView(CollocationAttributeFilterView):

    template_name = 'stats/ngram_bindings.html'
    form_class = NgramBindingForm
    ngram_regex = ur'(?:\w|-)'

    @classmethod
    def scores(cls):
        return [name[1:-6] for name in dir(cls) if name.endswith('_score')]

    def get_context_data(self, **kwargs):
        """Add nodes and links to the context"""
        context = super(NgramWordBindingDistributionView, self).get_context_data(**kwargs)
        form = self.form_class(self.request.POST or None)
        if form.is_valid():
            pos_tag = form.cleaned_data['pos_tag']
            score_func = getattr(self, '_' + form.cleaned_data['scoring_function'] + '_score')
            article_dict = self._populate_article_dict(pos_tag, score_func)
            context['avg_precision'] = self._caclculate_average_precision(article_dict)
            context['article_dict'] = [(key, sorted(values, key=lambda x: x[2], reverse=True))
                                       for key, values in article_dict.iteritems()]
        context['form'] = form
        return context

    def _populate_article_dict(self, pos_tag, score_func):
        article_dict = defaultdict(list)
        self.queryset = self.queryset.values_list('ngram', flat=True)
        if pos_tag:
            self.queryset = self.queryset.filter(_pos_tag__regex='^{0}$'.format(pos_tag))
        rel_ngram_set = set(self.queryset.filter(tags__is_relevant=True))
        irrel_ngram_set = set(self.queryset.filter(tags__is_relevant=False))
        for article in Article.objects.filter(cluster_id=self.queryset.model.CLUSTER_ID):
            text = article.stemmed_text
            for ngram in sorted(article.articlecollocation_set.values_list('ngram', flat=True),
                                key=lambda x: len(x[0].split())):
                if ngram in rel_ngram_set:
                    is_rel = True
                elif ngram in irrel_ngram_set:
                    is_rel = False
                else:
                    continue
                ngram_count = text.count(ngram)
                if ngram_count <= 2:
                    continue
                score = score_func(ngram, text, article_dict)
                article_dict[article].append((ngram, ngram_count, score, is_rel))
        return article_dict

    def _weight_prev_bigram_score(self, ngram, text, article_dict):
        w1, w2 = ngram.split()
        distribution_dict = Counter(re.findall(ur'{0} ({1}+)'.format(w1, self.ngram_regex), text))
        score = distribution_dict[w2] / sum(distribution_dict.values())
        return score

    def _weight_last_bigram_score(self, ngram, text, article_dict):
        w1, w2 = ngram.split()
        distribution_dict = Counter(re.findall(ur'({1}+) {0}'.format(w2, self.ngram_regex), text))
        score = distribution_dict[w1] / sum(distribution_dict.values())
        return score

    def _weight_both_bigram_score(self, ngram, text, article_dict):
        w1, w2 = ngram.split()
        distribution_dict = Counter(re.findall(ur'({1}+) {0}'.format(w2, self.ngram_regex), text))
        N1 = sum(distribution_dict.values())
        N1_len = len(distribution_dict)
        score = distribution_dict[w1] / N1
        distribution_dict = Counter(re.findall(ur'{0} ({1}+)'.format(w1, self.ngram_regex), text))
        N2 = sum(distribution_dict.values())
        N2_len = len(distribution_dict)
        score += distribution_dict[w2] / N2
        if N1_len == 1 or N2_len == 1 and text.count(ngram) > 5:
            score += 2
        return score / 2

    def _weight_ngram_score(self, ngram, text, article_dict):
        w1, w2 = ngram.split()
        distribution_dict = Counter(re.findall(ur'({1}+) {0}'.format(w2, self.ngram_regex), text))
        N1 = sum(distribution_dict.values())
        N1_len = len(distribution_dict)
        score = distribution_dict[w1] / N1
        distribution_dict = Counter(re.findall(ur'{0} ({1}+)'.format(w1, self.ngram_regex), text))
        N2 = sum(distribution_dict.values())
        N2_len = len(distribution_dict)
        score += distribution_dict[w2] / N2
        if N1_len == 1 or N2_len == 1 and text.count(ngram) > 5:
            score += 2
        return score / 2

    def _caclculate_average_precision(self, article_dict):
        avg_prec_list = []
        for k, v in article_dict.items():
            local_precision = []
            sorted_scores = sorted(v, key=lambda x: x[2], reverse=True)
            tp = len([x for x in sorted_scores if x[3]])
            if tp == 0:
                continue
            correct_count = 0
            for ngram, _, _, is_rel in sorted_scores:
                correct_count += int(is_rel)
                tp += int(not is_rel)
                local_precision.append(correct_count / tp)
                if int(local_precision[-1]) == 1:
                    break
            avg_prec_list.append(max(local_precision))
        return sum(avg_prec_list) / len(avg_prec_list)


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

        next_url = self.request.GET.get('next', reverse('testcollection_model',
                                        args=[self.model_name]))
        return HttpResponseRedirect(next_url)

    def get_context_data(self, **kwargs):
        """Add nodes and links to the context"""
        context = super(ClearCachedAttrView, self).get_context_data(**kwargs)
        context['model_name'] = self.model_name
        return context
