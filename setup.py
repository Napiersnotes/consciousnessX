#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
consciousnessX - Quantum-Biological AGI Framework
Setup configuration for package installation
"""

from setuptools import setup, find_packages
import os
import re

# Read version from src/__init__.py
def get_version():
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'src', '__init__.py'), 'r', encoding='utf-8') as f:
        version_file = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

# Read long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="consciousnessx",
    version=get_version(),
    author="Dafydd Napier",
    author_email="your.email@example.com",
    description="Simulates artificial consciousness using quantum gravity in microtubules",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Napiersnotes/consciousnessX",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    install_requires=requirements,
    extras_require={
        'quantum': ['qutip>=4.7.0', 'projectq>=0.6.0'],
        'bio': ['biopython>=1.80', 'neo>=0.12.0'],
        'hpc': ['mpi4py>=3.1.0', 'dask>=2023.3.0', 'ray>=2.3.0'],
        'visualization': ['pyvista>=0.38.0', 'plotly>=5.14.0', 'dash>=2.9.0'],
        'dev': [
            'pytest>=7.3.0',
            'pytest-cov>=4.0.0',
            'black>=23.3.0',
            'flake8>=6.0.0',
            'mypy>=1.3.0',
            'sphinx>=7.0.0',
        ],
        'all': [
            'qutip>=4.7.0',
            'projectq>=0.6.0',
            'biopython>=1.80',
            'neo>=0.12.0',
            'mpi4py>=3.1.0',
            'dask>=2023.3.0',
            'ray>=2.3.0',
            'pyvista>=0.38.0',
            'plotly>=5.14.0',
            'dash>=2.9.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'consciousnessx=src.cli.main:main',
            'cx-simulate=src.cli.simulate:main',
            'cx-visualize=src.cli.visualize:main',
        ],
    },
    include_package_data=True,
    package_data={
        'src': ['config/*.yaml', 'data/*.json', 'templates/*.html'],
    },
    keywords='consciousness quantum biology ai neuroscience microtubules penrose',
    project_urls={
        'Documentation': 'https://github.com/Napiersnotes/consciousnessX/wiki',
        'Source': 'https://github.com/Napiersnotes/consciousnessX',
        'Tracker': 'https://github.com/Napiersnotes/consciousnessX/issues',
    },
)
