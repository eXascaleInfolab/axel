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

                                                                          #  TP   FP   FN
CONFIG_PIPELINES = {'simple': {'rank_attr': 'log_prob', 'pipeline': []},  # (143, 225, 296)
                    'filter_NNP': {'rank_attr': 'log_prob',
                    'pipeline': ['is_not_proper_noun']},                  # (153, 212, 286)
                    'filter_NNP_digit': {'rank_attr': 'log_prob',
                    'pipeline': ['is_not_proper_noun', 'is_not_digit']}   # (155, 210, 284)
                    #  {'name': 'decay_progr_prob+NNP', 'rank_attr': '????', 'pipeline': []}
                    }


class NgramManager(models.Manager):
    """Custom manager that requests MS ngram service seamlessly and saves in to DB"""

    def get(self, *args, **kwargs):
        try:
            result = self.get_query_set().get(*args, **kwargs)
        except Ngram.DoesNotExist:
            if 'value' in kwargs:
                ngram = kwargs['value']
                log_prob = ms_ngram_service.GetJointProbability(ngram.encode('utf-8'))
                print ngram, log_prob
                result = Ngram.objects.create(value=ngram, log_prob=log_prob)
            else:
                raise
        return result


class Ngram(models.Model):
    """Describes ngram"""
    value = models.TextField()
    log_prob = models.FloatField()

    objects = NgramManager()

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
        # POS seq attribute need to be set before calling this function
        return 'NNP' not in self.pos_seq

    @property
    def is_not_digit(self):
        """Check if ngram contains digit"""
        return not re.search(r'\d', self.value)

    @classmethod
    def create_from_sentence(cls, sentence):
        """
        Generates all possible ngrams from a sentence and stores their probability in the DB.
        :param sentence: Sentence.
        """
        existing = set(Ngram.objects.values_list("value", flat=True))
        for sent in (sentence.sentence1_pos_seq, sentence.sentence2_pos_seq):
            for tokens in sent:
                for i in range(1, 6):
                    for ngram in nltk.ngrams(zip(*tokens)[0], i):
                        ngram = u' '.join(ngram).lower()
                        if ngram not in existing:
                            log_prob = ms_ngram_service.GetJointProbability(ngram.encode('utf-8'))
                            print ngram, log_prob
                            Ngram.objects.create(value=ngram, log_prob=log_prob)
                            existing.add(ngram)


class NgramWrapper(dict):
    """
    Dict wrapper that either retrieves objects from the provided dict or from Ngram model.
    """
    def __getitem__(self, key):
        """ return item from the dict, if not present check ngram model """
        key = key.lower()
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            return Ngram.objects.get(value=key)


class SentenceManager(models.Manager):
    """Normalize sentence on GET to not create duplicates"""

    def get(self, *args, **kwargs):
        if 'sentence1' in kwargs:
            if not kwargs['sentence1'].startswith('I '):
                kwargs['sentence1'] = kwargs['sentence1'][0].lower() + kwargs['sentence1'][1:]
        if 'sentence2' in kwargs:
            if not kwargs['sentence2'].startswith('I '):
                kwargs['sentence2'] = kwargs['sentence2'][0].lower() + kwargs['sentence2'][1:]

        return self.get_query_set().get(*args, **kwargs)


class Sentence(models.Model):
    sentence1 = models.TextField()
    sentence1_pos_seq = JSONField()
    sentence2 = models.TextField()
    sentence2_pos_seq = JSONField()

    objects = SentenceManager()

    def __unicode__(self):
        """String representation"""
        return self.sentence1

    class Meta:
        unique_together = ('sentence1', 'sentence2')

    @classmethod
    def _tokenize_pos_tags(cls, sentence):
        """
        Tokenize sentence and return lists of tokens.
        :return: list of POS tagged ngrams splitted on punctuation
         [[('this', 'DT'),
          ('puts', 'NNS'),
          ('the', 'DT'),
          ('calendar', 'NN'),
          ('into', 'IN')],
         [('My', 'NNP'), ('Calendars', 'NNP')],
         [('for', 'IN'), ('both', 'DT'), ('users', 'NNS')],
         [('calendars', 'NNS')]]
        """
        tokens = nltk.pos_tag(nltk.regexp_tokenize(sentence, nlp.Stemmer.TOKENIZE_REGEXP))
        tokens = [list(x[1]) for x in itertools.groupby(tokens, lambda y: Ngram.PUNKT_RE.match(y[0]))
                  if not x[0]]
        return tokens

    @classmethod
    def get_sentence_prob(cls, sentence_pos_seq):
        """Return averaged probability scores for the sentence using 2- to 5-ngram splits"""
        scores = defaultdict(list)
        for tokens in sentence_pos_seq:
            for i in range(1, 6):
                for ngram in nltk.ngrams(zip(*tokens)[0], i):
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
                positional_data[sentence.id] = pos_sent_data
        return positional_data

    def prob_sorted_ngrams(self, ngrams_all=None, config=CONFIG_PIPELINES['simple']):
        """Returns diverging ngrams in the sentence"""
        # TODO: Exclude 100 most frequent words?

        if ngrams_all is None:
            ngrams_all = NgramWrapper()

        # sentence can contain more than one sequence of tokens
        position_data = defaultdict(list)
        for group, tokens in enumerate(self.sentence1_pos_seq):
            for i in range(1, 6):
                for ngram_num, ngram_pos in enumerate(nltk.ngrams(tokens, i)):
                    ngram = ' '.join(zip(*ngram_pos)[0])
                    ngram_obj = ngrams_all[ngram]
                    ngram_obj.pos_seq = zip(*ngram_pos)[1]
                    # report ngram with position

                    rank_attr = getattr(ngram_obj, config['rank_attr'])

                    add = True
                    for pipeline_func in config['pipeline']:
                        if not getattr(ngram_obj, pipeline_func):
                            add = False
                            break

                    if add:
                        position_data[i].append({'position': (i, group, ngram_num),
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
                # We always get only 2 n-1 grams out n-gram.
                for i in range(2):
                    position_data[index-1][i+j]['dec_score'] += prob

        try:
            results = [sorted(position_data[2], key=lambda x: x['rank_attr'])[0]]
        except Exception, e:
            print self.sentence1
            print self.sentence2
            return
        return results

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
        true_edit_data = defaultdict(set)
        true_edit_data1 = defaultdict(list)
        for edit in cls.objects.all():
            true_edit_data[edit.sentence_id].add(tuple(edit.edit_data['orig']['serial']))
            true_edit_data1[edit.sentence_id].append(edit.edit_data['orig']['serial'])

        for sent_id, true_sent_edit_data in true_edit_data.iteritems():
            if len(true_sent_edit_data) != len(true_edit_data1[sent_id]):
                print sent_id

        for sent_id, true_sent_edit_data in true_edit_data.iteritems():
            if sent_id in position_data:
                sent_edit_data = position_data[sent_id]
                # -----DEBUG INFO - for printing FP
                debug_sent_edit_data = sent_edit_data[:]
                # ----- END DEBUG
                index = 0
                sent_tp_count = 0
                for ngram_len, group_num, ngram_position in sorted(true_sent_edit_data):
                    if index >= len(sent_edit_data):
                        fn += 1
                        continue

                    pred_len, pred_group, pred_pos = sent_edit_data[index]['position']
                    interval = [ngram_position, ngram_position + ngram_len - pred_len]

                    if group_num == pred_group and min(interval) <= pred_pos <= max(interval):
                        tp += 1
                        sent_tp_count += 1
                        del debug_sent_edit_data[index]
                        index += 1
                    else:
                        fn += 1
                fp += len(sent_edit_data) - sent_tp_count
                if debug_sent_edit_data:
                    sent = Sentence.objects.get(id=sent_id)
                    print sent.sentence1
                    print sent.sentence2
                    for x in debug_sent_edit_data:
                        print x
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
                    return (1, i, j), word
                j += 1
    edit_data = instance.edit_data
    """
    :type: dict
    """
    instance.edit_type = Edit.REPLACE

    unigram_data = get_edited_unigram(sentence.sentence1, edit_data['orig']['start_pos'], edit_data['orig']['end_pos'])
    if unigram_data:
        edit_data['orig']['serial'], edit_data['orig']['word'] = unigram_data
    else:
        edit_data['orig']['word'] = ''
        instance.edit_type = Edit.INSERT

    unigram_data = get_edited_unigram(sentence.sentence2, edit_data['new']['start_pos'], edit_data['new']['end_pos'])
    if unigram_data:
        edit_data['new']['serial'], edit_data['new']['word'] = unigram_data
    else:
        edit_data['new']['word'] = ''
        # trigram
        instance.edit_type = Edit.DELETE

    if instance.edit_type == Edit.INSERT:
        _, group, serial = edit_data['new']['serial']
        if serial == 0:
            edit_data['orig']['serial'] = (1, group, 0)
            edit_data['new']['serial'] = (2, group, 0)
        else:
            # TODO: need to check if insert was at the the end
            edit_data['orig']['serial'] = (2, group, serial-1)
            edit_data['new']['serial'] = (3, group, serial-1)

    if instance.edit_type == Edit.DELETE:
        _, group, serial = edit_data['orig']['serial']
        if serial == 0:
            edit_data['orig']['serial'] = (2, group, 0)
            edit_data['new']['serial'] = (1, group, 0)
        else:
            # TODO: need to check if insert was at the the end
            edit_data['orig']['serial'] = (3, group, serial-1)
            edit_data['new']['serial'] = (2, group, serial-1)
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

    instance.sentence1_pos_seq = Sentence._tokenize_pos_tags(instance.sentence1)
    instance.sentence2_pos_seq = Sentence._tokenize_pos_tags(instance.sentence2)
