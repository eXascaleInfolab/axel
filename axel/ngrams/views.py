from django.views.generic import ListView, DetailView

from axel.ngrams.models import Sentence


class SentenceList(ListView):
    """View to display paginated article list with available actions"""
    model = Sentence
    context_object_name = 'sentences'
    template_name = 'ngrams/sentence_list.html'
    paginate_by = 50


class SentenceDetailView(DetailView):
    """Display article details"""
    model = Sentence
    context_object_name = "sentence"
    template_name = 'ngram/sentence.html'

