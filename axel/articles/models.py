import json
import os

from django.conf import settings
from django.contrib.contenttypes import generic
from django.db import models
from django.db.models import F, Sum
from django.db.models.signals import pre_delete, post_save
from django.dispatch import receiver

from jsonfield import JSONField
from test_collection.models import TaggedCollection

from axel.articles.utils.db import db_cache
from axel.libs.utils import get_contexts, get_contexts_ngrams
from axel.stats.models import SWCollocations, Collocations
import axel.stats.scores as scores


class Venue(models.Model):
    """Describes article venue"""
    name = models.CharField(max_length=255)
    acronym = models.CharField(max_length=10)

    class Meta:
        """Meta info"""
        ordering = ['acronym']

    def __unicode__(self):
        """String representation"""
        return self.acronym


def pdf_upload_to(instance, filename):
    """
    Determines where to upload a PDF
    :type instance: Article
    :type filename: str
    """
    return '/'.join((instance.venue.acronym, str(instance.year), filename))


class Article(models.Model):
    """Main article model"""
    title = models.CharField(max_length=255, default='')
    abstract = models.TextField(default='')
    venue = models.ForeignKey(Venue)
    year = models.IntegerField()
    link = models.URLField(null=True)
    citations = models.IntegerField(default=0)
    pdf = models.FileField(upload_to=pdf_upload_to)
    stemmed_text = models.TextField(default='')
    text = models.TextField(default='')
    index = models.TextField(default='')
    index_nonstemmed = JSONField()
    cluster_id = models.CharField(max_length=255)

    class Meta:
        """Meta info"""
        ordering = ['-year']

    def __unicode__(self):
        """String representation"""
        return u"{0} {1}: {2}".format(self.venue, self.year, self.title.replace(',', ' '))

    @property
    def CollocationModel(self):
        """
        Get correct collocation model according to the desired split
        :rtype Collocation
        """
        return CLUSTERS_DICT[self.cluster_id]

    def dbpedia_graph(self, redirects=True):
        """
        Generate a dbpedia category TREE using networkx
        :rtype: nx.Graph
        """
        import tempfile
        import requests
        from networkx.readwrite import json_graph
        tmpdir = tempfile.gettempdir()
        if redirects:
            graph_object = tmpdir + '/' + str(self.id) + 'redirects.' + '.dbpedia.json'
        else:
            graph_object = tmpdir + '/' + str(self.id) + '.dbpedia.json'
        if not os.path.exists(graph_object):

            stop_uris_set = open(settings.ABS_PATH('stop_uri.txt')).read().split()
            stop_uris_set = set([x.split('/')[-1] for x in stop_uris_set])

            def recurse_populate_graph(resource, graph, depth):
                if resource in stop_uris_set:
                    return
                if depth == 0:
                    return
                if 'Category' in resource:
                    query = u'SELECT ?broader, ?related, ?broaderof WHERE' \
                            u' {{{{ <http://dbpedia.org/resource/{0}> skos:broader ?broader }}' \
                            u' UNION {{ ?broaderof skos:broader <http://dbpedia.org/resource/{0}> }}' \
                            u' UNION {{ ?related skos:related <http://dbpedia.org/resource/{0}> }}' \
                            u' UNION {{ <http://dbpedia.org/resource/{0}> skos:related ?related }}}}'.format(resource)

                    results = []
                    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
                    sparql.setReturnFormat(JSON)
                    sparql.setQuery(query)
                    results.extend(sparql.query().convert()['results']['bindings'])
                    for result in results:
                        for rel_type, value in result.iteritems():
                            uri = value['value']
                            parent_resource = uri.split('/')[-1]
                            #print '  ' * (3 - depth), resource, '->', parent_resource
                            graph.add_edge(resource, parent_resource, type=rel_type)
                            recurse_populate_graph(parent_resource, graph, depth-1)
                else:
                    if resource == 'cumulative gain':
                        resource = 'Discounted_cumulative_gain'
                    elif resource == 'world wide web conference':
                        resource = 'International_World_Wide_Web_Conference'
                    wiki_cat_query = u'http://en.wikipedia.org/w/api.php?action=query&titles={0}&prop=categories&cllimit=50&clshow=!hidden&format=json&redirects'
                    results = json.loads(requests.get(wiki_cat_query.format(resource)).text)['query']['pages'].values()[0]
                    if 'missing' in results:
                        results = json.loads(requests.get(wiki_cat_query.format(resource.title())).text)['query']['pages'].values()[0]
                        if 'missing' in results:
                            print results, resource
                            results = []
                        else:
                            results = [c['title'].replace(' ', '_') for c in results['categories']]
                    else:
                        results = [c['title'].replace(' ', '_') for c in results['categories']]
                    rel_type = "subject"
                    for parent_resource in results:
                        #print '  ' * (3 - depth), resource, '->', parent_resource
                        graph.add_edge(resource, parent_resource, type=rel_type)
                        recurse_populate_graph(parent_resource, graph, depth-1)

            import networkx as nx
            from SPARQLWrapper import SPARQLWrapper, JSON

            graph = nx.Graph()
            ngrams = set(self.articlecollocation_set.values_list('ngram', flat=True))
            ngrams = self.CollocationModel.COLLECTION_MODEL.objects.filter(ngram__in=ngrams)
            for ngram in ngrams:
                if 'dbpedia' in ngram.source or (redirects and 'wiki_redirect' in ngram.source):
                    recurse_populate_graph(ngram.ngram, graph, 2)

            json_graph.dump(graph, open(graph_object, 'w'))
        else:
            graph = json_graph.load(open(graph_object))
        # BELOW CODE RETURNS 2 max connected components
        # results = []
        # for component in nx.connected_components(graph):
        #     component = [node for node in component if 'Category' not in node]
        #     results.append(component)
        #
        # # select 2 max clusters
        # results.sort(key=lambda x: len(x), reverse=True)
        # return [item for sublist in results[:1] for item in sublist]
        return graph

    @property
    def wikilinks_graph(self):
        """
        Generate a wikilinks graph using networkx
        :rtype: Graph
        """
        import tempfile
        from networkx.readwrite import json_graph
        import networkx as nx
        import re
        import requests

        tmpdir = tempfile.gettempdir()
        graph_object = tmpdir + '/' + str(self.id) + '.wikilinks.json'

        def _get_links(ngram):
            ngram_links = json.loads(requests.get(template_query.format(ngram)).text)
            try:
                ngram_links = ngram_links['query']['pages'].values()[0]['links']
            except KeyError:
                return []
            ngram_links = [re.sub(r' \(.+\)', '', link['title'].lower()) for link in ngram_links]
            ngram_links = set([ngram for ngram in ngram_links if len(ngram.split()) > 1])
            return ngram_links

        if not os.path.exists(graph_object):
            graph = nx.Graph()
            links_dict = {}
            template_query = u'http://en.wikipedia.org/w/api.php?action=query&titles={0}&prop=links&plnamespace=0&pllimit=500&format=json'
            article_ngrams = list(self.articlecollocation_set.values_list('ngram', flat=True))
            for i, ngram1 in enumerate(article_ngrams):
                if ngram1 in links_dict:
                    ngram1_links = links_dict[ngram1]
                else:
                    ngram1_links = _get_links(ngram1)
                    links_dict[ngram1] = ngram1_links
                for j in range(i+1, len(article_ngrams)):
                    ngram2 = article_ngrams[j]
                    if ngram2 in links_dict:
                        ngram2_links = links_dict[ngram2]
                    else:
                        ngram2_links = _get_links(ngram2)
                        links_dict[ngram2] = ngram2_links
                    if ngram1 in ngram2_links or ngram2 in ngram1_links:
                        graph.add_edge(ngram1, ngram2)
            json_graph.dump(graph, open(graph_object, 'w'))
            return graph

        else:
            graph = json_graph.load(open(graph_object))
            return graph

    def _create_collocations(self):
        """Create collocation for the article"""
        from axel.libs import nlp
        if self.index and not self.testcollocations_set.exists():
            index = json.loads(self.index)
            # found collocs = found existing + found new
            collocs = nlp.collocations(index)

            # Create other collocations
            for name, score in collocs.iteritems():
                if score > 0:
                    TestCollocations.objects.create(ngram=name, article=self, count=score)

    @classmethod
    def create_collocations(cls, cluster_id):
        """
        Populates collocation for the specified article collection
        :param cluster_id: cluster id to specify article collection
        """
        print 'Initial population...'
        for article in cls.objects.filter(cluster_id=cluster_id):
            # create all found collocations inside single article
            article._create_collocations()
        # then rescan all given already existing
        all_collocs = set(TestCollocations.objects.values_list('ngram', flat=True))

        print 'Existing population...'
        # add existing if do not exist yet
        for article in cls.objects.filter(cluster_id=cluster_id):
            index = json.loads(article.index)
            for colloc in all_collocs.intersection(index.keys()):
                # get or create because we are not filtrating old ones
                TestCollocations.objects.get_or_create(ngram=colloc,
                                                         article=article,
                                                         defaults={'count': index[colloc]})

        # we could screw up counts completely, need to update them
        print 'Starting updates...'
        from axel.libs.utils import print_progress
        from axel.libs.nlp import _update_ngram_counts
        for article in print_progress(cls.objects.filter(cluster_id=cluster_id)):
            ngrams = sorted(article.testcollocations_set.values_list('ngram', 'count'),
                            key=lambda x: (x[1], x[0]))
            if not ngrams:
                continue
            new_ngrams = _update_ngram_counts([c.split() for c in zip(*ngrams)[0]],
                json.loads(article.index))
            new_ngrams = sorted(new_ngrams.items(), key=lambda x: (x[1], x[0]))
            new_ngrams = [k for k in new_ngrams if k[1] > 0]
            if new_ngrams != ngrams:
                obsolete_ngrams = set(ngrams).difference(new_ngrams)
                article.testcollocations_set.filter(ngram__in=zip(*obsolete_ngrams)[0]) \
                    .delete()
                for ngram, score in set(new_ngrams).difference(ngrams):
                    TestCollocations.objects.create(ngram=ngram, count=score, article=article)


class TestCollocations(models.Model):
    """
    Model contains collocation for each article and their respective counts,
    CAN BE DELETED AND REGENERATED.
    Exists for testing different collocations models, can be moved to ArticleCollocation afterwards.
    """
    ngram = models.CharField(max_length=255)
    count = models.IntegerField()
    article = models.ForeignKey(Article)


class ArticleCollocationsManager(models.Manager):

    def get_query_set(self):
        return super(ArticleCollocationsManager, self).get_query_set()\
            .filter(article__cluster_id=self.model.CLUSTER_ID)


class ArticleCollocation(models.Model):
    """
    Model contains ngram -- possible entity for the collection, decision choice
    and extra attributes.
    """
    ngram = models.CharField(max_length=255)
    count = models.IntegerField()
    # duplication to efficiently perform ordering
    total_count = models.IntegerField()
    article = models.ForeignKey(Article)
    tags = generic.GenericRelation(TaggedCollection, for_concrete_model=False)

    extra_fields = JSONField()

    # Populated by subclasses
    judged_data = None

    class Meta:
        """Meta info"""
        ordering = ['-total_count', '-count']
        unique_together = ('ngram', 'article')

    def __unicode__(self):
        """String representation"""
        return u"{0},{1}".format(self.ngram, self.article)

    @property
    def is_relevant(self):
        """
        Get relevance information.
        Used in article detail view.
        """
        try:
            return int(self.judged_data[unicode(self)])
        except KeyError:
            return -1

    @property
    @db_cache('extra_fields')
    def context(self):
        """
        Get random context for collocation, used in collocation list view,
        :rtype: unicode
        :returns: context if found, ngram itself otherwise
        """
        # prevent contexts from bigger ngrams
        bigger_ngrams = self.article.articlecollocation_set.filter(ngram__contains=self.ngram)\
            .exclude(ngram=self.ngram).values_list('ngram', flat=True)
        context = next(get_contexts(self.article.text, self.ngram, bigger_ngrams), self.ngram)
        return context

    def all_contexts(self, func=get_contexts):
        """
        Get all contexts for detailed view page
        :rtype: list
        :returns: contexts if found, [ngram] otherwise
        """
        contexts = []
        text = self.article.text
        bigger_ngrams = self.article.articlecollocation_set.filter(ngram__contains=self.ngram)\
            .exclude(ngram=self.ngram).values_list('ngram', flat=True)
        for context in func(text, self.ngram, bigger_ngrams):
            contexts.append(context)
        return contexts

    def all_contexts_pos(self, func=get_contexts):
        """
        Get all contexts for part-of-speech tagging (Do not exclude bigger n-grams)
        :rtype: list
        :returns: contexts if found, [ngram] otherwise
        """
        contexts = []
        text = self.article.text
        for context in func(text, self.ngram, []):
            contexts.append(context)
        return contexts

    @property
    @db_cache('extra_fields')
    def pos_tag(self):
        """
        Defines part-of-speech tag for ngram
        :return: Part-of-Speech tag
        :rtype: unicode
        """
        return scores.pos_tag(self.ngram, self.all_contexts_pos(func=get_contexts_ngrams))

    @property
    @db_cache('extra_fields')
    def pos_tag_prev(self):
        """
        Retrieves part-of-speech tag for the word before ngram
        :return: dict of Part-of-Speech tags with scores
        :rtype: dict
        """
        return scores.pos_tag_pos(self.ngram, self.all_contexts_pos(func=get_contexts_ngrams))

    @property
    @db_cache('extra_fields')
    def pos_tag_after(self):
        """
        Retrieves part-of-speech tag for the word before ngram
        :return: dict of Part-of-Speech tags with scores
        :rtype: dict
        """
        return scores.pos_tag_pos(self.ngram, self.all_contexts_pos(func=get_contexts_ngrams), tag_pos=1)

    @classmethod
    def scores(cls):
        result = []
        for method in dir(cls):
            if method.endswith('_score'):
                result.append(method)
        return result


class CSArticleCollocations(ArticleCollocation):
    CLUSTER_ID = 'CS_COLLOCS'
    COLLECTION_MODEL = Collocations
    objects = ArticleCollocationsManager()
    judged_data = dict([line.rsplit(',', 1) for line in open(settings.ABS_PATH('CSArticleCollocations.csv')).read().split('\n')])
    maxent_judged_data = dict([line.rsplit(',', 1) for line in open(settings.ABS_PATH('maxent_CSArticleCollocations.csv')).read().split('\n')])
    judged_data.update(maxent_judged_data)

    class Meta:
        proxy = True


class SWArticleCollocations(ArticleCollocation):
    CLUSTER_ID = 'SW_COLLOCS'
    COLLECTION_MODEL = SWCollocations
    objects = ArticleCollocationsManager()
    judged_data = dict([line.rsplit(',', 1) for line in open(settings.ABS_PATH('SWArticleCollocations.csv')).read().split('\n')])

    class Meta:
        proxy = True

    def __unicode__(self):
        """String representation"""
        return u"{0},{1}".format(self.ngram, os.path.split(self.article.pdf.name)[-1][:-4])


class Author(models.Model):
    """Basic author model"""
    name = models.CharField(max_length=255)
    first_name = models.CharField(max_length=50, null=True, blank=True)
    last_name = models.CharField(max_length=50, null=True, blank=True)
    middle_name = models.CharField(max_length=50, null=True, blank=True)

    class Meta:
        ordering = ['name']

    def __unicode__(self):
        """String representation"""
        return self.name


class ArticleAuthor(models.Model):
    """Relationship of the author to the article"""
    article = models.ForeignKey(Article)
    author = models.ForeignKey(Author)

    def __unicode__(self):
        return u'{0}: {1}'.format(self.author, self.article)


@receiver(pre_delete, sender=ArticleCollocation)
def clean_collocations(sender, instance, **kwargs):
    """
    Reduce collocation count on delete for ArticleCollocation
    :type instance: ArticleCollocation
    """
    colloc = instance.article.CollocationModel.objects.get(ngram=instance.ngram)
    colloc.count -= instance.count
    colloc.save()


@receiver(pre_delete, sender=Article)
def clean_pdf(sender, instance, **kwargs):
    """
    Remove PDF on deletion
    :type instance: Article
    """
    os.unlink(instance.pdf.path)


@receiver(post_save, sender=ArticleCollocation)
def update_global_collocations(sender, instance, created, **kwargs):
    """
    Increment collocation count on create for ArticleCollocation
    :type instance: ArticleCollocation
    """
    if kwargs.get('raw'):
        return
    if created:
        colloc, created_local = instance.article.CollocationModel.objects.get_or_create(
            ngram=instance.ngram, defaults={'count': instance.count})
        if not created_local:
            colloc.count = F('count') + instance.count
            colloc.save()
    else:
        # Recalculate count otherwise
        colloc = instance.COLLECTION_MODEL.objects.get(ngram=instance.ngram)
        colloc.count = sender.objects.filter(ngram=instance.ngram).aggregate(count=Sum('count'))['count']
        colloc.save()


#@receiver(post_save, sender=Article)
#def create_acronyms(sender, instance, created, **kwargs):
#    """
#    Add acronyms and their disambiguations on create
#    :type instance: Article
#    """
#    from axel.stats.models import Collocations
#    from axel.articles.utils import nlp
#    if instance.stemmed_text and not ArticleCollocation.objects.filter(article=instance).exists():
#        text = instance.stemmed_text
#        acronyms = nlp.acronyms(text)
#        for abbr, name in collocs:
#            acolloc, created = ArticleCollocation.objects.get_or_create(ngram=name,
#                article=instance, defaults={'count': score})
#            if not created:
#                acolloc.score = score
#                acolloc.save()
#            colloc, created = Collocations.objects.get_or_create(ngram=name)
#            if not created:
#                colloc.count = F('count') + 1
#                colloc.save()


CLUSTERS_DICT = dict([(model.CLUSTER_ID, model) for model in (CSArticleCollocations, SWArticleCollocations)])
