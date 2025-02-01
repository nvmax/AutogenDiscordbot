from setuptools import setup, find_packages

setup(
    name="autogen-bot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'discord.py>=2.3.2',
        'python-dotenv>=1.0.0',
        'chromadb>=0.4.22',
        'sentence-transformers>=2.2.2',
        'requests>=2.31.0',
        'python-decouple>=3.8',
    ],
)
