import json
import os

from django.conf import settings
from django.db import models
from django.db.models import F, Sum
from django.db.models.signals import pre_delete, post_save
from django.dispatch import receiver

from haystack.query import SearchQuerySet


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
    cluster_id = models.CharField(max_length=255)

    class Meta:
        """Meta info"""
        ordering = ['-year']

    def __unicode__(self):
        """String representation"""
        return u"{0} {1}: {2}".format(self.venue, self.year, self.title)

    @property
    def CollocationModel(self):
        """
        Get correct collocation model according to the desired split
        :rtype Collocation
        """
        from axel.stats.models import CLUSTERS_DICT
        return CLUSTERS_DICT[self.cluster_id]

    @property
    def dbpedia_graph(self):
        """
        Generate a dbpedia category TREE using networkx
        :rtype: list
        """
        if settings.BUILD_DBPEDIA_GRAPHS:

            def recurse_populate_graph(resource, graph):
                if 'Category' in resource:
                    query = u'SELECT ?broader WHERE {{ <http://dbpedia.org/resource/{0}> skos:broader ?broader }}'.format(resource)
                    attr = 'broader'
                else:
                    url_resource = resource.capitalize().replace(' ', '_')
                    query = u'SELECT ?subject WHERE {{ <http://dbpedia.org/resource/{0}> dcterms:subject ?subject }}'.format(url_resource)
                    attr = 'subject'

                sparql.setQuery(query)
                results = sparql.query().convert()
                results = results['results']['bindings']
                if attr == 'broader':
                    results = results[:1]
                for result in results:
                    uri = result[attr]['value']
                    parent_resource = uri.split('/')[-1]
                    if parent_resource in graph:
                        graph.add_edge(resource, parent_resource)
                        return
                    graph.add_edge(resource, parent_resource)
                    recurse_populate_graph(parent_resource, graph)

            import networkx as nx
            from SPARQLWrapper import SPARQLWrapper, JSON
            sparql = SPARQLWrapper("http://dbpedia.org/sparql")
            sparql.setReturnFormat(JSON)

            graph = nx.Graph()
            ngrams = set(self.articlecollocation_set.values_list('ngram', flat=True))
            ngrams = self.CollocationModel.objects.filter(ngram__in=ngrams)
            for ngram in ngrams:
                if ngram.source == 'dbpedia':
                    recurse_populate_graph(ngram.ngram, graph)
            results = []
            for component in nx.connected_components(graph):
                component = [node for node in component if 'Category' not in node]
                if len(component) > 2:
                    results.extend(component)

            return results
        return None

    def create_collocations(self):
        """Create collocation for the article"""
        from axel.articles.utils import nlp
        if self.index and not self.articlecollocation_set.exists():
            index = json.loads(self.index)
            # found collocs = found existing + found new
            collocs = nlp.collocations(index)
            # all previously existing collocs
            all_collocs = set(self.CollocationModel.objects.values_list('ngram', flat=True))
            # get all new
            new_collocs = set(collocs.keys()).difference(all_collocs)

            # get all existing not found, (those that have score <= 2)
            # we do not need to check <=2 condition here, is should be automatically satisfied
            for colloc in all_collocs.difference(collocs.keys()).intersection(index.keys()):
                ArticleCollocation.objects.get_or_create(ngram=colloc,
                    article=self, defaults={'count': index[colloc]})

            # Create other collocations
            for name, score in collocs.iteritems():
                if score > 0:
                    acolloc, created = ArticleCollocation.objects.get_or_create(ngram=name,
                        article=self, defaults={'count': score})
                    if not created:
                        acolloc.score = score
                        acolloc.save()

            # Scan existing articles for new collocations
            for colloc in new_collocs:
                new_articles = SearchQuerySet().filter(content__exact=colloc)\
                .exclude(id='articles.article.'+str(self.id)).values_list('id', flat=True)
                new_articles = set([a_id.split('.')[-1] for a_id in new_articles])
                for article in Article.objects.filter(id__in=new_articles):
                    if article.cluster_id != self.cluster_id:
                        continue
                    index = json.loads(article.index)
                    # Check that collocation is in index and
                    # second check that we don't already have bigger collocations
                    # This is incorrect to run alone, we need a full update after new collocations
                    # found!!!!!!!
                    if colloc in index:
                        correct_count = index[colloc] - int(article.articlecollocation_set.filter(
                            ngram__contains=colloc).aggregate(count=Sum('count'))['count'] or 0)
                        if correct_count > 0:
                            ArticleCollocation.objects.create(ngram=colloc,
                                article=article, count=correct_count)


class ArticleCollocation(models.Model):
    """Model contains collocation for each article and their count"""
    ngram = models.CharField(max_length=255)
    count = models.IntegerField()
    article = models.ForeignKey(Article)

    class Meta:
        """Meta info"""
        ordering = ['-count']
        unique_together = ('ngram', 'article')

    def __unicode__(self):
        """String representation"""
        return u"{0}: {1}".format(self.article, self.ngram)

    @property
    def is_relevant(self):
        """
        Get relevance information from underlying Collocation model.
        Used in article detail view.
        """
        cModel = self.article.CollocationModel
        try:
            return cModel.objects.get(ngram=self.ngram).tags.all()[0].is_relevant
        except (cModel.DoesNotExist, IndexError):
            return -1


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
    if created:
        colloc, created_local = instance.article.CollocationModel.objects.get_or_create(
            ngram=instance.ngram, defaults={'count': instance.count})
        if not created_local:
            colloc.count = F('count') + instance.count
            colloc.save()
    else:
        # Recalculate count otherwise
        colloc = instance.article.CollocationModel.objects.get(ngram=instance.ngram)
        colloc.count = sender.objects.filter(ngram=instance.ngram).aggregate(count=Sum('count'))['count']
        colloc.save()


@receiver(post_save, sender=Article)
def create_collocations(sender, instance, **kwargs):
    """
    Add collocations on create
    :type instance: Article
    """
    instance.create_collocations()


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
