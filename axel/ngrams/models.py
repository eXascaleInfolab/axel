# coding=utf-8
from __future__ import division

from collections import defaultdict
import nltk
import re
import itertools
import MicrosoftNgram
from jsonfield import JSONField
from nltk.metrics.association import BigramAssocMeasures as bigram_assoc

from django.db.models.signals import pre_save
from django.dispatch import receiver
from django.db import models

from axel.libs import nlp

ms_ngram_service = MicrosoftNgram.LookupService('37a80cca-9fee-487f-9bbd-c45f252534df',
                                                'bing-body/apr10/5')

                                                                          #  TP   FP   TN
CONFIG_PIPELINES = {'simple': {'rank_attr': 'log_prob', 'pipeline': []},  # (122, 206, 273)
                    'filter_NNP': {'rank_attr': 'log_prob',
                    'pipeline': ['is_not_proper_noun']},                  # (131, 195, 264)
                    #  {'name': 'decay_progr_prob+NNP', 'rank_attr': '????', 'pipeline': []}
                    }


class Ngram(models.Model):
    """Describes ngram"""
    value = models.TextField()
    log_prob = models.FloatField()

    PUNKT_RE = re.compile(r'[`~/%\*\+\[\]\.?!,":;()\'|]+')

    class Meta:
        """Meta info"""
        ordering = ['value']

    def __unicode__(self):
        """String representation"""
        return self.value

    @property
    def is_not_proper_noun(self):
        """Checks if ngram contains proper noun in it"""
        return 'NNP' not in self.pos_seq

    @property
    def is_not_digit(self):
        """Check if ngram contains digit"""
        return re.search(r'\d', self.value)

    @classmethod
    def create_from_sentence(cls, pos_tag_sents):
        """
        :param pos_tag_sents: sentence to parse, in JSON POS-tagged format, for example:
                              [('I', 'PRP'), ('have', 'VBP'), ('a', 'DT'), ('dog', 'NN')].
        """
        existing = set(Ngram.objects.values_list("value", flat=True))
        # join ngrams with tags
        pos_tag_sents = ['/'.join(x) for x in pos_tag_sents]
        pos_tag_sents = [list(x[1]) for x in itertools.groupby(pos_tag_sents,
                                                lambda x: cls.PUNKT_RE.match(x)) if not x[0]]
        for pos_tag_sent in pos_tag_sents:
            for i in range(1, 6):
                for pos_ngram in nltk.ngrams(pos_tag_sent, i):
                    ngram = zip(*[x.rsplit('/', 1) for x in pos_ngram])[0]
                    ngram = u' '.join(ngram).lower()
                    if ngram not in existing:
                        log_prob = ms_ngram_service.GetJointProbability(ngram.encode('utf-8'))
                        print ngram, log_prob
                        Ngram.objects.create(value=ngram, log_prob=log_prob)
                        existing.add(ngram)

    def get_POS_tag(self, sentence, position):
        pass


class NgramWrapper(dict):
    """
    Dict wrapper that either retrieves objects from the provided dict or from Ngram model.
    """
    def __getitem__(self, key):
        """ return item from the dict, if not present check ngram model """
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            return Ngram.objects.get(value=key)


class Sentence(models.Model):
    sentence1 = models.TextField()
    sentence1_pos_seq = models.CharField(max_length=255)
    sentence2 = models.TextField()
    sentence2_pos_seq = models.CharField(max_length=255)

    def __unicode__(self):
        """String representation"""
        return self.sentence1

    @classmethod
    def _tokenize(cls, sentence):
        """Tokenize sentence and return lists of tokens."""
        tokens = nltk.regexp_tokenize(sentence, nlp.Stemmer.TOKENIZE_REGEXP)
        tokens = [list(x[1]) for x in itertools.groupby(tokens, lambda y: Ngram.PUNKT_RE.match(y))
                  if not x[0]]
        return tokens

    @classmethod
    def get_sentence_prob(cls, sentence):
        """Return averaged probability scores for the sentence using 2- to 5-ngram splits"""
        scores = defaultdict(list)
        for tokens in Sentence._tokenize(sentence):
            for i in range(1, 6):
                for ngram in nltk.ngrams(tokens, i):
                    log_prob = Ngram.objects.get(value=' '.join(ngram)).log_prob
                    scores[i].append(10**log_prob)
        return dict([(i, sum(probs)/len(probs) if probs else 0) for i, probs in
                     sorted(scores.items(), key=lambda x: x[0])])

    @classmethod
    def get_positional_metrics_data(cls, config):
        """Gets positional data to calculate metrics"""
        positional_data = {}
        ngram_dict = NgramWrapper([(ngram.value, ngram) for ngram in Ngram.objects.all()])
        for sentence in cls.objects.all():
            pos_sent_data = sentence.prob_sorted_ngrams(ngram_dict, config)
            if pos_sent_data:
                positional_data[sentence.id] = [pos_sent_data]
        return positional_data

    def prob_sorted_ngrams(self, ngrams_all=None, config=CONFIG_PIPELINES['filter_NNP']):
        """Returns diverging ngrams in the sentence"""
        # TODO: Exclude 100 most frequent words?

        if ngrams_all is None:
            ngrams_all = NgramWrapper()

        # sentence can contain more than one sequence of tokens
        position_data = defaultdict(list)
        for tokens in Sentence._tokenize_positions(self.sentence1):
            for i in range(1, 6):
                for ngram_pos in nltk.ngrams(tokens, i):
                    ngram = ' '.join(zip(*ngram_pos)[0])
                    ngram_obj = ngrams_all[ngram]
                    # report ngram with position

                    rank_attr = getattr(ngram_obj, config['rank_attr'])

                    add = True
                    for pipeline_func in config['pipeline']:
                        if not getattr(ngram_obj, pipeline_func):
                            add = False
                            break

                    if add:
                        position_data[i].append({'position': ngram_pos,
                                                 'ngram': ngram,
                                                 'rank_attr': rank_attr})

        # add decaying probability to everything
        for ngrams in position_data.values():
            prob = -1
            for ngram_dict in sorted(ngrams, key=lambda x: x['rank_attr']):
                ngram_dict['dec_score'] = prob
                prob /= 2

        # update n-1 ngrams with decaying prob from high-order n-grams
        index_order = sorted(position_data.keys())

        # ngram are ordered in a sentence order
        for index in index_order[1:]:
            for j, ngram_dict in enumerate(position_data[index]):
                prob = ngram_dict['dec_score']
                for i, _ in enumerate(nltk.ngrams(zip(*ngram_dict['position'])[0], index-1)):
                    position_data[index-1][i+j]['dec_score'] += prob

        try:
            lowest_bigram = sorted(position_data[2], key=lambda x: x['rank_attr'])[0]['position']
        except:
            return
        return (lowest_bigram[0][1], lowest_bigram[1][2])

    def small_likelihood_ratio(self, ngram_obj):
        """
        Tries to identify errors using likelihood ration test under binomial distribution assumption
        """
        # TODO: here
        bigram_assoc.likelihood_ratio()


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
    edit_data = JSONField()
    sentence = models.ForeignKey(Sentence)
    edit1 = models.CharField(max_length=255)
    # Edit2 is not null when type is REPLACE
    edit2 = models.CharField(max_length=255, null=True)

    def __unicode__(self):
        orig_ngram = self.edit_data['orig']['word']
        new_ngram = self.edit_data['new']['word']
        return orig_ngram + u' → ' + new_ngram

    @classmethod
    def _tokenize_positions(cls, sentence):
        """
        Tokenize sentence and return lists of tokens with corresponding positions in the sentence.
        """
        positional_tokens = []
        tokenize_regex = re.compile(nlp.Stemmer.TOKENIZE_REGEXP)
        for match in tokenize_regex.finditer(sentence):
            positional_tokens.append((match.group(), match.start(), match.end()))

        tokens = [list(x[1]) for x in
                  itertools.groupby(positional_tokens, lambda y: Ngram.PUNKT_RE.match(y[0]))
                  if not x[0]]
        return tokens

    @classmethod
    def calculate_positional_metrics(cls, position_data):
        """
        Calculate precision and recall given the positional data for
        possible incorrect places in the data.
        :param position_data: of form {sentence_id: {1: [(GROUP_NUM, NGRAM_NUM), ...], ...}, ...}
        :type position_data: dict
        """
        tp = 0
        fp = 0
        fn = 0
        temp_edit_data = cls.objects.values_list('sentence', 'start_pos_orig', 'end_pos_orig')
        true_edit_data = defaultdict(set)
        true_edit_data1 = defaultdict(list)
        for sent_id, start_pos, end_pos in temp_edit_data:
            true_edit_data[sent_id].add((start_pos, end_pos))
            true_edit_data1[sent_id].append((start_pos, end_pos))

        for sent_id, true_sent_edit_data in true_edit_data.iteritems():
            if len(true_sent_edit_data) != len(true_edit_data1[sent_id]):
                print sent_id

        for sent_id, true_sent_edit_data in true_edit_data.iteritems():
            if sent_id in position_data:
                sent_edit_data = sorted(position_data[sent_id])
                # -----DEBUG INFO - for printing FP
                debug_sent_edit_data = sent_edit_data[:]
                # ----- END DEBUG
                index = 0
                sent_tp_count = 0
                for true_start_pos, true_end_pos in sorted(true_sent_edit_data):
                    if index >= len(sent_edit_data):
                        fn += 1
                    elif true_start_pos >= sent_edit_data[index][0] and true_end_pos <= sent_edit_data[index][1]:
                        tp += 1
                        sent_tp_count += 1
                        del debug_sent_edit_data[index]
                        index += 1
                    else:
                        fn += 1
                fp += len(sent_edit_data) - sent_tp_count
                if debug_sent_edit_data:
                    sent = Sentence.objects.get(id=sent_id).sentence1
                    print sent
                    for start_pos, end_pos in debug_sent_edit_data:
                        print sent[start_pos:end_pos], start_pos, end_pos
                    print
            else:
                fn += len(true_sent_edit_data)
        return tp, fp, fn

    @classmethod
    def calculate_final_metrics(cls, edit_data):
        """Calculate precision and recall given the edit data"""
        orig_edit_data = set(cls.objects.values_list())
        precision = len(edit_data - orig_edit_data) / edit_data
        recall = len(orig_edit_data - edit_data) / edit_data

        return precision, recall


@receiver(pre_save, sender=Edit)
def populate_extra_fields(sender=None, instance=None, **kwargs):
    """
    :type instance: Edit
    """
    sentence = instance.sentence

    def get_edited_unigram(sentence, start_pos, end_pos):
        for i, part in enumerate(Edit._tokenize_positions(sentence)):
            j = 0
            for word, w_start, w_end in part:
                if w_start <= start_pos and w_end >= end_pos:
                    return i, j, word
                j += 1
    edit_data = instance.edit_data
    unigram_data = get_edited_unigram(sentence.sentence1, edit_data['orig']['start_pos'], edit_data['orig']['end_pos'])
    if unigram_data:
        edit_data['orig']['group'], edit_data['orig']['serial'], edit_data['orig']['word'] = unigram_data
    else:
        edit_data['orig']['word'] = ''

    unigram_data = get_edited_unigram(sentence.sentence2, edit_data['new']['start_pos'], edit_data['new']['end_pos'])
    if unigram_data:
        edit_data['new']['group'], edit_data['new']['serial'], edit_data['new']['word'] = unigram_data
    else:
        edit_data['new']['word'] = ''
    instance.edit_data = edit_data


@receiver(pre_save, sender=Sentence)
def extra_sentence_normalization(sender=None, instance=None, **kwargs):
    """
    :type instance: Sentence
    """
    # lowercase first letter to not pos tag improperly
    if not instance.sentence1.startswith('I '):
        instance.sentence1 = instance.sentence1[0].lower() + instance.sentence1[1:]
    if not instance.sentence2.startswith('I '):
        instance.sentence2 = instance.sentence2[0].lower() + instance.sentence2[1:]
