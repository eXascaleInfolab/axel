from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from axel.articles.utils.concepts_index import update_index


class Collocations(models.Model):
    """Aggregated collocation statistics model"""
    keywords = models.CharField(max_length=255)
    count = models.IntegerField(default=1)

    class Meta:
        """Meta info"""
        ordering = ['-count']

    def __unicode__(self):
        """String representation"""
        return "{0}".format(self.keywords)



@receiver(post_save, sender=Collocations)
def create_collocations(sender, instance, created, **kwargs):
    """
    Add to index on create
    :type instance: Collocations
    """
    if created:
        update_index(instance.id, instance.keywords)

