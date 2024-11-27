from setuptools import setup, find_packages

setup(
    name="spatio-textual",
    version="0.1.0",
    description="A library for spatial textual analysis with a focus on the Corpus of Lake District Writing and Holocaust survivors' testimonies.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Ignatius Ezeani",
    author_email="igezeani@yahoo.com",
    url="https://github.com/yourusername/my_holocaust_analysis",
    packages=find_packages(),
    install_requires=[
        "spacy",
        "transformers"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)