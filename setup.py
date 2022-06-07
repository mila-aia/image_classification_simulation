from setuptools import setup, find_packages


setup(
    name='image_classification_simulation',
    version='0.0.1',
    packages=find_packages(include=[
        'image_classification_simulation', 'image_classification_simulation.*']),
    python_requires='>=3.8',
    install_requires=[
        'flake8',
        'flake8-docstrings',
        'gitpython',
        'tqdm',
        'jupyter',
        'mlflow==1.15.0',
        'orion>=0.1.14',
        'pyyaml>=5.3',
        'pytest>=4.6',
        'pytest-cov',
        'sphinx',
        'sphinx-autoapi',
        'sphinx-rtd-theme',
        'sphinxcontrib-napoleon',
        'sphinxcontrib-katex',
        'recommonmark',
        'protobuf==3.20.1',
        'datasets==2.2.2',
        'transformers==4.19.2',
        'torch==1.11.0', 'pytorch_lightning==1.2.7'],
    entry_points={
        'console_scripts': [
            'main=image_classification_simulation.main:main'
        ],
    }
)
