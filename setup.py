#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: setup.py.py
@time: 2024/3/3 10:46
"""

from setuptools import setup, find_packages
import sys
from pathlib import Path

if sys.version_info < (3, 7):
    sys.exit('biollm requires Python >= 3.7')

setup(
    name='BioLLM',
    version='0.1.2',
    description='A Standardized Framework for Integrating and Benchmarking Single-Cell Foundation Models.',
    long_description=Path('README.md').read_text('utf-8'),
    long_description_content_type="text/markdown",
    url='https://github.com/qiupinghust/BioLLM',
    author='Ping Qiu',
    author_email='qiuping1@genomics.cn',
    python_requires='>=3.7',
    install_requires=[
        l.strip() for l in Path('requirements.txt').read_text('utf-8').splitlines()
    ],
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
)
