import sys
from setuptools import setup


setup(name='fiesta',
      version='0.1',
      description='Automatic text preprocessing modul',
      url='https://github.com/kammillam/fiesta',
      author='Kamilla Mansurova',
      packages=['fiesta',
                'fiesta.automatic',
                'fiesta.preprocessing',
                'fiesta.feature_extraction',
                'fiesta.feature_selection',
                'fiesta.transformers',
                'fiesta.external_packages'],
      install_requires=[
          'scikit-learn',
          'pandas',
          'matplotlib',
          'nltk',
          'pyphen'
      ],
      dependency_links=[
            ],

      zip_safe=False)