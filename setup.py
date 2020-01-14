"""
Use setuptools to build a distributable package.
"""

import setuptools

with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

setuptools.setup(
    name="imgtxtgen",
    version="0.0.1",
    author="Harikrishnan Balagopal",
    author_email="harikrishmenon@gmail.com",
    description="A package to generate both images and text using GANs.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/HarikrishnanBalagopal/image-and-text-generation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
