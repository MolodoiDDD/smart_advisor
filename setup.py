from setuptools import setup, find_packages

setup(
    name="smart_advisor",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'chromadb>=0.4.0',
        'sentence-transformers>=2.2.0',
        'pymupdf>=1.22.0',
        'beautifulsoup4>=4.12.0',
        'python-dotenv>=1.0.0',
    ],
)