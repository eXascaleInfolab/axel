"""Unit-tests for articles app"""
import os
from django.test import TestCase
from django.conf import settings
from django.core.files import File
from axel.articles.models import Article
from axel.stats.models import Collocations


class CollocationsTest(TestCase):
    """Tests correct collocation extraction and deletion"""

    def test_create_article(self):
        """Test article creation and deletion"""
        article = Article(venue_id=3, year=1999)
        full_path = os.path.join(settings.ROOT_DIR, 'articles', 'fixtures',
            'Hofmann-SIGIR99.pdf')
        with open(full_path, 'rb') as pdf:
            article.pdf.save(os.path.basename(full_path), File(pdf), save=True)
        article.save()

        collocs = Collocations.objects.all()
        self.assertTrue(collocs)
        self.assertEqual(collocs[0].keywords, 'probabilistic latent semantic indexing')
        article.delete()

        # Check it's empty now
        collocs = Collocations.objects.filter(count__gt=0).exists()
        self.assertFalse(collocs)

