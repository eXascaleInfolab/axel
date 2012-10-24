"""Forms for the articles application"""
from django import forms


class PDFUploadForm(forms.Form):
    article_pdf = forms.FileField()

    def get_collocations(self):
        """Extract collocations using the self.cleaned_data dictionary"""
        pass
