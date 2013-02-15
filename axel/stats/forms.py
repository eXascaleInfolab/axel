from django import forms


class ScoreCacheResetForm(forms.Form):
    attr = forms.ChoiceField(choices=())

    def __init__(self, model_class, *args, **kwargs):
        super(ScoreCacheResetForm, self).__init__(*args, **kwargs)
        self.fields['attr'].choices = [(score, score) for score in model_class.CACHED_FIELDS]

