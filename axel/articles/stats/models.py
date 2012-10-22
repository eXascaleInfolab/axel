from django.db import models


class Collocations(models.Model):
    """Main article model"""
    keywords = models.CharField(max_length=255)
    count = models.IntegerField()

    class Meta:
        """Meta info"""
        ordering = ['-count']

    def __unicode__(self):
        """String representation"""
        return "{0}".format(self.keywords)
