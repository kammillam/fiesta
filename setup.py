import sys
from setuptools import setup


setup(name='fiesta',
      version='0.1',
      description='Automatic text preprocessing modul',
      url='https://github.com/kammillam/fiesta',
      author='Kamilla Mansurova',
      packages=['fiesta'],
      install_requires=[
          'scikit-learn',
          'pandas',
          'matplotlib',
          'nltk',
          'pyphen'
      ],
      dependency_links=[
            'https://github.com/WZBSocialScienceCenter/germalemma.git'
        ],

      zip_safe=False)