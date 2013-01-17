import json
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
    index = models.TextField(default='')

    class Meta:
        """Meta info"""
        ordering = ['-year']

    def __unicode__(self):
        """String representation"""
        return "{0} {1}: {2}".format(self.venue, self.year, self.title)

    @property
    def collocations(self):
        """Get co-locations from the saved stemmed text"""
        colocs = list(self.articlecollocation_set.values_list('count', 'keywords'))
        colocs.sort(key=lambda col: col[0], reverse=True)
        return colocs

    @property
    def CollocationModel(self):
        """
        Get correct collocation model according to the desired split
        :rtype CommonCollocationInfo
        """
        from axel.stats.models import Collocations, SWCollocations
        if self.venue.acronym == 'ARXIV':
            return SWCollocations
        else:
            return Collocations

    def create_collocations(self):
        """Create collocation for the article"""
        from axel.articles.utils import nlp
        if self.index and not self.articlecollocation_set.exists():
            index = json.loads(self.index)
            # found collocs = found existing + found new
            collocs = nlp.collocations(index)
            # all previously existing collocs
            all_collocs = set(self.CollocationModel.objects.values_list('keywords', flat=True))
            # get all new
            new_collocs = set(collocs.keys()).difference(all_collocs)

            # get all existing not found, (those that have score <= 2)
            # we do not need to check <=2 condition here, is should be automatically satisfied
            for colloc in all_collocs.difference(collocs.keys()).intersection(index.keys()):
                ArticleCollocation.objects.get_or_create(keywords=colloc,
                    article=self, defaults={'count': index[colloc]})

            # Create other collocations
            for name, score in collocs.iteritems():
                if score > 0:
                    acolloc, created = ArticleCollocation.objects.get_or_create(keywords=name,
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
                    index = json.loads(article.index)
                    # Check that collocation is in index and
                    # second check that we don't already have bigger collocations
                    # This is incorrect to run alone, we need a full update after new collocations
                    # found!!!!!!!
                    if colloc in index:
                        correct_count = index[colloc] - int(article.articlecollocation_set.filter(
                            keywords__contains=colloc).aggregate(count=Sum('count'))['count'] or 0)
                        if correct_count > 0:
                            ArticleCollocation.objects.create(keywords=colloc,
                                article=article, count=correct_count)


class ArticleCollocation(models.Model):
    """Model contains collocation for each article and their count"""
    keywords = models.CharField(max_length=255)
    count = models.IntegerField()
    article = models.ForeignKey(Article)

    class Meta:
        """Meta info"""
        ordering = ['-count']
        unique_together = ('keywords', 'article')

    def __unicode__(self):
        """String representation"""
        return "{0}: {1}".format(self.article, self.keywords)


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
        return "{0}: {1}".format(self.author, self.article)


@receiver(pre_delete, sender=ArticleCollocation)
def clean_collocations(sender, instance, **kwargs):
    """
    Reduce collocation count on delete for ArticleCollocation
    :type instance: ArticleCollocation
    """
    colloc = instance.article.CollocationModel.objects.get(keywords=instance.keywords)
    colloc.count -= instance.count
    colloc.save()


@receiver(post_save, sender=ArticleCollocation)
def update_global_collocations(sender, instance, created, **kwargs):
    """
    Increment collocation count on create for ArticleCollocation
    :type instance: ArticleCollocation
    """
    if created:
        colloc, created_local = instance.article.CollocationModel.objects.get_or_create(
            keywords=instance.keywords, defaults={'count': instance.count})
        if not created_local:
            colloc.count = F('count') + instance.count
            colloc.save()
    else:
        # Recalculate count otherwise
        colloc = instance.article.CollocationModel.objects.get(keywords=instance.keywords)
        colloc.count = sender.objects.filter(keywords=instance.keywords).aggregate(count=Sum('count'))['count']
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
#            acolloc, created = ArticleCollocation.objects.get_or_create(keywords=name,
#                article=instance, defaults={'count': score})
#            if not created:
#                acolloc.score = score
#                acolloc.save()
#            colloc, created = Collocations.objects.get_or_create(keywords=name)
#            if not created:
#                colloc.count = F('count') + 1
#                colloc.save()
