"""Forms for the articles application"""
import os
import subprocess

from django import forms
from axel.libs import nlp

import tempfile
from axel.libs.nlp import Stemmer
from axel.libs.parse_pdfx_xml import parse_pdfx_xml


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

    STEM_CHOICES = Stemmer.get_method_names()

    article_pdf = forms.FileField(label="Article PDF")
    stem_func = forms.ChoiceField(choices=STEM_CHOICES, label="Stemming function")

    def get_collocations(self):
        """Extract collocations using the self.cleaned_data dictionary"""
        full_name = handle_uploaded_file(self.cleaned_data['article_pdf'])
        stem_func = getattr(Stemmer, self.cleaned_data['stem_func'])

        if not os.path.exists(full_name + "x.xml"):
            subprocess.call(["pdfx", full_name])
        extracted_data = parse_pdfx_xml(full_name + "x.xml")

        full_text = nlp.get_full_text(extracted_data)['text']
        collocs = nlp.collocations(nlp.build_ngram_index(stem_func(full_text))).items()
        # order colocations
        collocs.sort(key=lambda col: col[1], reverse=True)
        return collocs


class ConceptAutocompleteForm(forms.Form):
    """Provide concept autocomplete field"""
    query = forms.CharField(widget=forms.TextInput(attrs={'class': 'input-xlarge search-query',
                                                          'autocomplete': 'off'}))
