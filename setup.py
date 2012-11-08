from setuptools import setup, find_packages
import sys, os

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.rst')).read()
NEWS = open(os.path.join(here, 'NEWS.txt')).read()


version = '0.9.4'

install_requires =\
  [ 'progressbar==2.2'
  , 'cmdln==1.1.2'
  , 'updatedir>=0.1'
  , 'cherrypy>=3.1.2'
  , 'pexpect==2.4'
  , 'numpy>=1.6.1'
  , 'scipy>=0.9.0'
  , 'tables==2.3.1' # setuptools doesn't seem to realize this is installed?
  ]


setup(name='hydrat',
    version=version,
    description="Classifier comparison framework",
    long_description=README + '\n\n' + NEWS,
    classifiers=\
    [ "Development Status :: 3 - Alpha"
    , "Environment :: Console"
    , "Intended Audience :: Science/Research"
    , "License :: OSI Approved :: BSD License"
    , "Operating System :: POSIX :: Linux"
    , "Programming Language :: Python"
    , "Topic :: Utilities"
    ],
    keywords='machinelearning textclassification documentprocessing',
    author='Marco Lui',
    author_email='saffsd@gmail.com',
    url='http://hydrat.googlecode.com',
    license='BSD',
    packages=find_packages('src'),
    package_dir = {'': 'src'},include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    entry_points={
        'console_scripts':
            ['hydrat=hydrat:main']
    }
)
