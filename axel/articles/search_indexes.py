import datetime

from haystack import indexes
from django.template import loader, Context

from axel.articles.models import Article
from axel.articles.utils import nlp
from axel.articles.utils.pdfcleaner import PDFCleaner


class ArticleIndex(indexes.RealTimeSearchIndex, indexes.Indexable):
    """Article indexer for haystack"""
    text = indexes.CharField(document=True, use_template=True)
    abstract = indexes.CharField(model_attr='abstract')
    pub_year = indexes.IntegerField(model_attr='year')

    def get_model(self):
        """returns underlying model"""
        return Article

    def index_queryset(self):
        """Used when the entire index for model is updated."""
        return self.get_model().objects.filter(year__lte=datetime.datetime.now())

    def should_update(self, instance, **kwargs):
        """Check if we are in a raw mode"""
        if kwargs.get('raw'):
            return False
        return True

    def prepare(self, obj):
        """
        Extract PDF contents and meta-data
        :type obj: Article
        """
        data = super(ArticleIndex, self).prepare(obj)

        # This could also be a regular Python open() call, a StringIO instance
        # or the result of opening a URL. Note that due to a library limitation
        # file_obj must have a .name attribute even if you need to set one
        # manually before calling extract_file_contents:
        obj.pdf.open()
        extracted_data = self._get_backend(None).extract_file_contents(obj.pdf.file)
        obj.pdf.close()
        result = nlp.stem_text(extracted_data['contents'])
        obj.stemmed_text = result['text']

        if result['abstract']:
            obj.abstract = result['abstract']
        if result['title']:
            obj.title = result['title']
        # save raw because we don't want to trigger signal again
        obj.save_base(raw=True)

        data['text'] = result['text']
        return data

