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
        # TODO: render same page with collocations included in the context
        return super(PDFCollocationsView, self).form_valid(form)