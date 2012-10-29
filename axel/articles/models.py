from django.db import models
from django.db.models import F


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
    title = models.CharField(max_length=255, null=True, blank=True)
    abstract = models.TextField(default='')
    venue = models.ForeignKey(Venue)
    year = models.IntegerField()
    link = models.URLField()
    citations = models.IntegerField(default=0)
    pdf = models.FileField(upload_to=pdf_upload_to)
    _stemmed_text = models.TextField(default='')

    class Meta:
        """Meta info"""
        ordering = ['-year']

    def __unicode__(self):
        """String representation"""
        return "{0} {1}: {2}".format(self.venue, self.year, self.title)

    @property
    def stemmed_text(self):
        """Just return inner field"""
        return self._stemmed_text

    @stemmed_text.setter
    def stemmed_text(self, text):
        """
        Save without triggering signal. We don't want to trigger it here.
        """
        from axel.articles.utils import nlp
        from axel.stats.models import Collocations
        self._stemmed_text = text
        # TODO: django 1.5 add update_fields attribute
        self.save_base(raw=True)
        # Do collocation processing after save
        collocs = nlp.collocations(text.lower())
        for score, name in collocs:
            colloc, created = Collocations.objects.get_or_create(keywords=name)
            if not created:
                colloc.count = F('count') + 1
                colloc.save()


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

