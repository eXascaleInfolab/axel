
# TODO: make a blog post out of a technique
class db_cache(object):
    """
    Decorator to cache the expensively computed field in some other field
    that supports dict-like assignment
    """

    def __init__(self, model_field):
        """
        :param model_field: field of the model to store and retrieve field from
        """
        self.model_field = model_field

    def __call__(self, f):
        def wrapper(object):
            fields = getattr(object, self.model_field)
            if f.__name__ in fields:
                return fields[f.__name__]
            else:
                value = f(object)
                fields[f.__name__] = value
                setattr(object, self.model_field, fields)
                object.save_base(raw=True)
                return value
        return wrapper


def db_cache_simple(func):
    """Simply populates/updates underscored field after assignment, otherwise calls func"""
    def wrapper(self):
        value = getattr(self, '_' + func.__name__)
        if value:
            return value
        else:
            value = func(self)
            setattr(self, '_' + func.__name__, value)
            self.save_base(raw=True)
            return value
    return wrapper
