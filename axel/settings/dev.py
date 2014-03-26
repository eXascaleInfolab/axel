from .base import *

DEBUG = True
TEMPLATE_DEBUG = DEBUG

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': 'dev.db',
        }
}

CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'unique-axel',
        'OPTIONS': {
            'MAX_ENTRIES': 100000
        }
    }
}

EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'

INTERNAL_IPS = ('',)
# set up Django Debug Toolbar if installed
try:
    import debug_toolbar
    MIDDLEWARE_CLASSES += ('debug_toolbar.middleware.DebugToolbarMiddleware', )
    INSTALLED_APPS += ('debug_toolbar', )
    DEBUG_TOOLBAR_CONFIG = {
        'INTERCEPT_REDIRECTS': False,
        'SHOW_TOOLBAR_CALLBACK': 'axel.libs.show_toolbar',
        'SQL_COUNT_THRESHOLD': 100,
    }
except ImportError:
    pass

ALLOWED_HOSTS = ['*']
PDFX_PATH = '/Users/dragoon/Libraries/pdfx/pdfx'
