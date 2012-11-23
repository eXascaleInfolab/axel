import numpy

from django.views.generic import TemplateView

from axel.stats.models import Collocations


class CollocationStats(TemplateView):
    """Main conceptual search view"""

    template_name = "stats/collocations.html"

    def get_context_data(self, **kwargs):
        """Add form to context"""
        context = super(CollocationStats, self).get_context_data(**kwargs)
        counts, bins = numpy.histogram(Collocations.objects.values_list('count', flat=True),
            bins=10)
        counts = [x+1 for x in counts]
        context['histogram_data'] = str(zip(bins, counts)).replace('(', '[').replace(')', ']')
        context['collocations'] =Collocations.objects.order_by('-count').values_list('count',
            'keywords')[:10]
        return context
