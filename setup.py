#! /usr/bin/env python
#

DESCRIPTION = "pyifu: Python library to manipulate Integral Field Unit cubes "
LONG_DESCRIPTION = """  Python library to manipulate Integral Field Unit (IFU) cubes """

DISTNAME = 'pyifu'
AUTHOR = 'Mickael Rigault'
MAINTAINER = 'Mickael Rigault' 
MAINTAINER_EMAIL = 'm.rigault@ipnl.in2p3.fr'
URL = 'https://github.com/MickaelRigault/pyifu'
LICENSE = 'Apache 2.0'
DOWNLOAD_URL = 'https://github.com/MickaelRigault/pyifu/0.6'
VERSION = '0.6.3'

try:
    from setuptools import setup, find_packages
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

def check_dependencies():
   install_requires = []

   # Just make sure dependencies exist, I haven't rigorously
   # tested what the minimal versions that will work are
   # (help on that would be awesome)
   try:
       import astropy
   except ImportError:
       install_requires.append('propobject')
    
   return install_requires

if __name__ == "__main__":

    install_requires = check_dependencies()

    if _has_setuptools:
        packages = find_packages()
        print(packages)
    else:
        # This should be updated if new submodules are added
        packages = ['pyifu']

    setup(name=DISTNAME,
          author=AUTHOR,
          author_email=MAINTAINER_EMAIL,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          install_requires=install_requires,
          packages=packages,
          package_data={},
          classifiers=[
              'Intended Audience :: Science/Research',
              'Programming Language :: Python :: 2.7',
              'Programming Language :: Python :: 3.5',              
              'License :: OSI Approved :: BSD License',
              'Topic :: Scientific/Engineering :: Astronomy',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS'],
      )
