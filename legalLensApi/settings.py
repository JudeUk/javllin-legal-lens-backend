"""
Django settings for legalLensApi project.

Generated by 'django-admin startproject' using Django 5.0.3.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/5.0/ref/settings/
"""
import ast
import os
from dotenv import load_dotenv

import dj_database_url

# Load environment variables from .env file
load_dotenv()


from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
# SECRET_KEY = 'django-insecure-9^(!u&t9f2l_6$)n)v%g3f7k=sy(7=(k4tp0118&1azttc1ur#'

SECRET_KEY = os.getenv('SECRET_KEY')

# Database configuration
# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.postgresql',
#         'NAME': os.getenv('DB_NAME'),
#         'USER': os.getenv('DB_USER'),
#         'PASSWORD': os.getenv('DB_PASSWORD'),
#         'HOST': os.getenv('DB_HOST'),
#         'PORT': os.getenv('DB_PORT'),
#     }
# }

MONGODB_URI = os.getenv('DATABASE_URL', 'mongodb+srv://javallin:javallin@javallinmongodb.par0ddp.mongodb.net/?retryWrites=true&w=majority&appName=JavallinMongoDb')

API_URL = os.getenv('API_URL')

HUGGINGFACE = os.getenv('HUGGINGFACE')


# DATABASES = {
#         "default": dj_database_url.parse(os.environ.get("DATABASE_URL")),
#     }

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.dummy',  # Use a dummy engine since MongoDB is NoSQL
        'NAME': 'dummy',  # Dummy name, not used for MongoDB
        'USER': '',  # Dummy user, not used for MongoDB
        'PASSWORD': '',  # Dummy password, not used for MongoDB
        'HOST': MONGODB_URI,
        'PORT': '',  # Port, if any, should be specified in the URI
    }
}


# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

# ALLOWED_HOSTS = ['127.0.0.1', '[::1]']
# ALLOWED_HOSTS = ast.literal_eval(os.getenv('ALLOWED_HOSTS', '[]'))

ALLOWED_HOSTS = ['*']

# CORS_ALLOWED_ORIGINS = ast.literal_eval(os.getenv('CORS_ALLOWED_ORIGINS', '[]'))

# CORS_ALLOWED_ORIGINS = os.getenv('CORS_ALLOWED_ORIGINS', 'https://javallin-frontend.vercel.app').split(',')

# CORS_ALLOWED_ORIGINS = 'https://javallin-frontend.vercel.app'



CORS_ORIGIN_ALLOW_ALL = True

CORS_ALLOW_CREDENTIALS = True







# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'caseApp',
    'corsheaders',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'django.middleware.common.CommonMiddleware',
]

ROOT_URLCONF = 'legalLensApi.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'legalLensApi.wsgi.application'


# Database
# https://docs.djangoproject.com/en/5.0/ref/settings/#databases

# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.sqlite3',
#         'NAME': BASE_DIR / 'db.sqlite3',
#     }
# }


# Password validation
# https://docs.djangoproject.com/en/5.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.0/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.0/howto/static-files/

STATIC_URL = 'static/'

# Default primary key field type
# https://docs.djangoproject.com/en/5.0/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


#CELERY SETTINGS

CELERY_BROKER_URL = ''
CELERYY_ACCEPT_CONTENT = ['application/json']
CELERY_RESULT_SERIALIZER='json'
CELERY_TASK_SERIRALIZER='json'
