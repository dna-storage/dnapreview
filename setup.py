# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
with open('LICENSE') as f:
    license = f.read()

setuptools.setup(
    name='dnapreview',
    version='0.9.2',
    author='James M. Tuck',
    description='dnapreview implements support for preview operations on dna storage.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email='jamesmtuck@gmail.com',
    url="https://github.com/dna-storage/dnastorage",
    license=license,
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Lesser GNU General Public License v2.1 (LGPLv2.1)",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
)
