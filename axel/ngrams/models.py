from django.db import models
import MicrosoftNgram
import nltk
import re
import itertools

from axel.libs import nlp

ms_ngram_service = MicrosoftNgram.LookupService('37a80cca-9fee-487f-9bbd-c45f252534df',
                                                'bing-body/apr10/5')


class Ngram(models.Model):
    """Describes ngram"""
    value = models.TextField()
    log_prob = models.FloatField()
    pos_seq = models.CharField(max_length=255)

    _PUNKT_RE = re.compile(r'[`~%\*\+\[\]\.?!,":;()\'|]+')

    class Meta:
        """Meta info"""
        ordering = ['value']

    def __unicode__(self):
        """String representation"""
        return self.value

    @classmethod
    def create_from_text(cls, text):
        existing = set(Ngram.objects.values_list("value", flat=True))
        for sentence in re.split(r'[.?!]', text):
            sentence = sentence.strip()
            if not sentence:
                continue
            pos_tag_sents = nltk.pos_tag(nltk.regexp_tokenize(sentence, nlp.Stemmer.TOKENIZE_REGEXP))
            # join ngrams with tags
            pos_tag_sents = ['/'.join(x) for x in pos_tag_sents]
            pos_tag_sents = [list(x[1]) for x in itertools.groupby(pos_tag_sents,
                                                lambda x: cls._PUNKT_RE.match(x)) if not x[0]]
            for pos_tag_sent in pos_tag_sents:
                for i in range(2, 6):
                    for pos_ngram in nltk.ngrams(pos_tag_sent, i):
                        ngram, pos_seq = zip(*[x.split('/') for x in pos_ngram])
                        ngram = u' '.join(ngram)
                        pos_seq = u' '.join(pos_seq)
                        if ngram not in existing:
                            log_prob = ms_ngram_service.GetJointProbability(ngram)
                            print ngram, log_prob, pos_seq
                            Ngram.objects.create(value=ngram, log_prob=log_prob, pos_seq=pos_seq)


class Sentence(models.Model):
    sentence1 = models.TextField()
    sentence2 = models.TextField()


class Edit(models.Model):
    DELETE = 'DEL'
    INSERT = 'INS'
    REPLACE = 'REP'

    EDIT_TYPES = (
        (DELETE, 'delete'),
        (INSERT, 'insert'),
        (REPLACE, 'replace'),
    )
    edit_type = models.CharField(max_length=3, choices=EDIT_TYPES)
    sentence = models.ForeignKey(Sentence)
    edit1 = models.CharField(max_length=255)
    # Edit2 is not null when type is REPLACE
    edit2 = models.CharField(max_length=255, null=True)
