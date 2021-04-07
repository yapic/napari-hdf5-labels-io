#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
from setuptools import setup, find_packages


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return codecs.open(file_path, encoding='utf-8').read()


# Add your dependencies in requirements.txt
# Note: you can add test-specific requirements in tox.ini
requirements = []
with open('requirements.txt') as f:
    for line in f:
        stripped = line.split("#")[0].strip()
        if len(stripped) > 0:
            requirements.append(stripped)


# https://github.com/pypa/setuptools_scm
# Currently incompatible with PyPi
# use_scm = {"write_to": "napari_hdf5_labels_io/_version.py"}

def local_scheme(version):
    return ""

setup(
    name='napari-hdf5-labels-io',
    author='Duway Nicolas Lesmes Leon',
    author_email='dlesmesleon@hotmail.com',
    license='GNU GPL v3.0',
    url='https://github.com/yapic/napari-hdf5-labels-io',
    description='Napari plugin to store set of layers in a .h5 file. Label layer are stored in a sparse representation',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='<3.9',
    install_requires=requirements,
    use_scm_version={"local_scheme": local_scheme},
    setup_requires=['setuptools_scm'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Framework :: napari',
        'Topic :: Software Development :: Testing',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
#        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ],
    entry_points={
        'napari.plugin': [
            'napari-hdf5-labels-io = napari_hdf5_labels_io',
        ],
    },
)
