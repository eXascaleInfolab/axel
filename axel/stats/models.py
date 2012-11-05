from django.db import models


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

