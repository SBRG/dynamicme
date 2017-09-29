try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='dynamicME',
      version='0.01',
      description='Time-course simulation of biomass, fluxes, macromolecules',
      author='Laurence Yang',
      author_email='lyang@eng.ucsd.edu',
      url='https://github.com/SBRG/dynamicme',
      packages=['dynamicme'],
      )
