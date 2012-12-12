from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from axel.articles.utils.concepts_index import update_index


class CommonCollocationInfo(models.Model):
    """Aggregated collocation statistics model"""
    keywords = models.CharField(max_length=255)
    count = models.IntegerField(default=1)

    # boolean field to mark concept either correct or not,
    # null when unknown
    correct = models.NullBooleanField(blank=True)

    class Meta:
        """Meta info"""
        abstract = True
        ordering = ['-count']

    def __unicode__(self):
        """String representation"""
        return "{0}".format(self.keywords)


class Collocations(CommonCollocationInfo):
    """Aggregated collocation statistics model for Computer Science"""


class SWCollocations(CommonCollocationInfo):
    """
    collocation for ScienceWISE
    everything is the same except table name
    """


@receiver(post_save, sender=Collocations)
def create_collocations(sender, instance, created, **kwargs):
    """
    Add to index on create
    :type instance: Collocations
    """
    if created:
        update_index(instance.id, instance.keywords)

