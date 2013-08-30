from django.core.urlresolvers import reverse
from django.http import Http404, HttpResponseRedirect
from django.shortcuts import render_to_response, get_object_or_404
from django.template import RequestContext
from django.views.decorators.http import require_POST
from django.views.generic.edit import FormView
from django.views.generic import ListView, DetailView, TemplateView

from axel.articles.forms import PDFUploadForm, ConceptAutocompleteForm
from axel.articles.models import Article
from axel.libs.mixins import JSONResponseMixin
from axel.stats.models import Collocations


class PDFCollocationsView(FormView):
    """Extract and display collocations from pdf document"""
    template_name = 'articles/pdfcollocations.html'
    form_class = PDFUploadForm

    def form_valid(self, form):
        """
        Do valid form post-processing, extract collocations
        :type form: PDFUploadForm
        """
        collocs = form.get_collocations()
        return self.render_to_response(self.get_context_data(form=form, collocations=collocs))


class ConceptualSearchView(TemplateView):
    """Main conceptual search view"""

    template_name = "articles/adv_search.html"

    def get_context_data(self, **kwargs):
        """Add form to context"""
        context = super(ConceptualSearchView, self).get_context_data(**kwargs)
        context['form'] = ConceptAutocompleteForm
        return context


class ArticleList(ListView):
    """View to display paginated article list with available actions"""
    model = Article
    context_object_name = 'articles'
    template_name = 'articles/article_list.html'
    paginate_by = 50


class ArticleDetailView(DetailView):
    """Display article details"""
    model = Article
    context_object_name = "article"
    template_name = 'articles/article.html'


class ConceptAutocompleteView(JSONResponseMixin, TemplateView):
    """View to retrieve autocomplete concept search results"""

    def render_to_response(self, context):
        form = ConceptAutocompleteForm(self.request.GET)
        if form.is_valid():
            query = form.cleaned_data['query']
            # search concepts
            results = Collocations.objects.filter(ngram__icontains=query)
            results = [colloc.ngram+' '+str(colloc.count) for colloc in results]
            context = {'results': results}
            return JSONResponseMixin.render_to_response(self, context)
        raise Http404


@require_POST
def filter_articles_view(request):
    """View that shows filtered articles based on concepts"""
    concepts = request.POST.getlist('concepts')
    articles = Article.objects.filter(articlecollocation__ngram__in=concepts).distinct()
    return render_to_response('partial/list.html', {'articles': articles},
                        context_instance=RequestContext(request))
