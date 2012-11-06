from django.db import models
from django.db.models import F
from django.db.models.signals import pre_delete, post_save
from django.dispatch import receiver


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

    class Meta:
        """Meta info"""
        ordering = ['-year']
        unique_together = ('title', 'year', 'venue')

    def __unicode__(self):
        """String representation"""
        return "{0} {1}: {2}".format(self.venue, self.year, self.title)

    @property
    def collocations(self):
        """Get co-locations from the saved stemmed text"""
        from axel.articles.utils import nlp
        colocs = nlp.collocations(self.stemmed_text)
        colocs.sort(key=lambda col: col[0], reverse=True)
        return colocs


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



@receiver(pre_delete, sender=Article)
def clean_collocations(sender, instance, **kwargs):
    """
    Reduce collocation count on delete
    :type instance: Article
    """
    from axel.stats.models import Collocations
    collocations = ArticleCollocation.objects.filter(article=instance).values_list('keywords',
                                                                                    flat=True)
    Collocations.objects.filter(keywords__in=collocations).update(count=(F('count') - 1))


@receiver(post_save, sender=Article)
def create_collocations(sender, instance, created, **kwargs):
    """
    Add collocations on create
    :type instance: Article
    """
    from axel.stats.models import Collocations
    from axel.articles.utils import nlp
    if instance.stemmed_text and not ArticleCollocation.objects.filter(article=instance).exists():
        text = instance.stemmed_text
        collocs = nlp.collocations(text)
        for score, name in collocs:
            acolloc, created = ArticleCollocation.objects.get_or_create(keywords=name,
                article=instance, defaults={'count': score})
            if not created:
                acolloc.score = score
                acolloc.save()
            colloc, created = Collocations.objects.get_or_create(keywords=name)
            if not created:
                colloc.count = F('count') + 1
                colloc.save()
