#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
#from wsipipe_isyntax import __version__
__version__ = '0.1.0'

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = ['Click>=7.0', ]

setup_requirements = [ ]

test_requirements = [ ]

project = "wsipipe_isyntax"

setup(
    author="David Morrison, Christina Fell",
    author_email='dm236@st-andrews.ac.uk, cmf21@st-andrews.ac.uk',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    description="An extension to add isyntax processing to wsipipe.",
    entry_points={
        'console_scripts': [
            'wsipipe_isyntax=wsipipe_isyntax.cli:main',
        ],
    },
    install_requires=['wsipipe'],
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='wsipipe_isyntax',
    name='wsipipe_isyntax',
    packages=find_packages(include=['wsipipe_isyntax']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/StAndrewsMedTech/wsipipe_isyntax',
    version=__version__,
    zip_safe=False,
)
