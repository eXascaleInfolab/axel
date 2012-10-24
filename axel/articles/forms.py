"""Forms for the articles application"""
from django import forms
from django.template import loader, Context
from haystack import connections
from axel.articles.utils import nlp
from axel.articles.utils.pdfcleaner import PDFCleaner

import tempfile


def handle_uploaded_file(f):
    """
    Put uploaded file to the tmp folder
    :returns: full saved file path
    """
    temp_name = tempfile.gettempdir() + f.name
    with open(temp_name, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return temp_name


class PDFUploadForm(forms.Form):
    """Handle uploaded PDF and extract co-locations"""
    article_pdf = forms.FileField(label="Article PDF")

    def get_collocations(self):
        """Extract collocations using the self.cleaned_data dictionary"""
        full_name = handle_uploaded_file(self.cleaned_data['article_pdf'])
        collocs = []
        with open(full_name) as pdf_obj:
            extracted_data = connections['default'].get_backend().extract_file_contents(pdf_obj)
            result = PDFCleaner.clean_pdf_data(extracted_data['contents'])
            t = loader.select_template(('search/indexes/articles/article_text.txt', ))
            full_text = t.render(Context({'extracted': result}))
            full_text = nlp.lemmatize(full_text)
            collocs = nlp.collocations(full_text.lower())
        return collocs
