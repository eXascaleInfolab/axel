from django.views.generic.edit import FormView
from django.views.generic import ListView, DetailView

from axel.articles.forms import PDFUploadForm
from axel.articles.models import Article


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
