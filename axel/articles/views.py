from django.views.generic.edit import FormView

from axel.articles.forms import PDFUploadForm


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
