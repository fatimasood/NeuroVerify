"""
Setup script for Trustify package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="trustify",
    version="1.0.0",
    author="Bharati et al.",
    author_email="goswami.rakesh@gmail.com",
    description="Fuzzy Logic-Based Hybrid Framework for Detecting Fake News",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/trustify-replication",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "nltk>=3.8.0",
        "transformers>=4.30.0",
        "torch-geometric>=2.3.0",
        "networkx>=3.1",
        "pillow>=10.0.0",
    ],
)