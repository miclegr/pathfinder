from setuptools import setup, find_packages

VERSION = "0.0"
NAME = 'pathfinder'
DESCRIPTION = "JAX implementation of Pathfinder"
MAINTAINER = 'Michele Gregori'
MAINTAINER_EMAIL = 'michelegregorits@gmail.com'
LICENSE = 'Apache 2.0'
DOWNLOAD_URL = 'https://github.com/miclegr/pathfinder.git'
URL = 'https://github.com/miclegr/pathfinder'

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=open('README.md').read(),
      license=LICENSE,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      install_requires=['numpy>=1.18', 'jax>=0.2.24'],
      packages=find_packages(),
      )
