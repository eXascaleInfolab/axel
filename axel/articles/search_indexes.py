import datetime
import json

from haystack import indexes

from axel.articles.models import Article
from axel.articles.utils import nlp


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
        return self.get_model().objects.filter(year__lte=datetime.datetime.now().year)

    def should_update(self, instance, **kwargs):
        """Check if we are in a raw mode"""
        if kwargs.get('raw') and not kwargs.get('created'):
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
        result = nlp.get_full_text(extracted_data['contents'])
        # get rid of multiple whitespaces
        obj.stemmed_text = ' '.join(result['text'].split())
        obj.index = json.dumps(nlp.build_ngram_index(nlp.Stemmer.stem_wordnet(obj.stemmed_text)))

        if result['abstract']:
            obj.abstract = result['abstract']
        if result['title']:
            obj.title = result['title']
        # save raw because we don't want to trigger signal again
        obj.save_base(raw=True)

        data['text'] = obj.stemmed_text
        return data

