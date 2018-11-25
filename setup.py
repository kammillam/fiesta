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
      ],
      zip_safe=False)