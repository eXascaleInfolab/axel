"""Match extracted collocation with DBPedia entities"""
from django.contrib.contenttypes.models import ContentType
from django.core.management.base import BaseCommand, CommandError
from django.db.models import get_model

from optparse import make_option
from lxml import etree
import requests
from suds.client import Client
from suds.xsd.doctor import Import, ImportDoctor
from test_collection.models import TaggedCollection
from axel.libs.utils import print_progress


DBLP_URL = 'http://dblp.l3s.de/WS/aspl2.php?wsdl'
imp = Import('http://schemas.xmlsoap.org/soap/encoding/',
             location='http://schemas.xmlsoap.org/soap/encoding/')
dblp_client = Client(DBLP_URL,plugins=[ImportDoctor(imp)])

DBPEDIA_REQ = u'http://lookup.dbpedia.org/api/search' \
              u'.asmx/KeywordSearch?QueryClass=&QueryString={0}&MaxHits=1'



class Command(BaseCommand):
    option_list = BaseCommand.option_list + (
        make_option('--model', '-m',
            action='store',
            dest='model',
            help='model to match'),
        )
    help = 'Produce match between collocation and DBPeadia concepts'

    def handle(self, *args, **options):
        results = []
        model = options['model']
        if not model:
            raise CommandError("need to specify model")

        # get model
        app_label, model = options['model'].split('.')[1::2]
        Model = get_model(app_label, model)
        ct = ContentType.objects.get_for_model(Model)

        tagged_objects = set(TaggedCollection.objects.filter(content_type=ct).values_list(
            'object_id', flat=True))

        for obj in print_progress(Model.objects.all()):
            object = obj
            """:type: Collocation"""
            if object.id in tagged_objects:
                continue

            # perform search using dbpedia
#            r = requests.get(DBPEDIA_REQ.format(object.ngram))
#            xml = etree.fromstring(r.text.replace('encoding="utf-8"',''))
#            result = xml.find('.//{http://lookup.dbpedia.org/}Label')
#            desc = xml.find('.//{http://lookup.dbpedia.org/}Description')
#            if result is not None and desc is not None and result.text.lower() == object.ngram:
#                results.append(object.ngram)

            # perform keyword search from dblp
            dblp_res = dblp_client.service.all_keywords_year(searchTerm=object.ngram,startYear=1999,
                endYear=2012,limit=1)
            if dblp_res and dblp_res[0].keyword.lower() == object.ngram:
                print object.ngram
                results.append(object.ngram)


        print results
        print len(results)

