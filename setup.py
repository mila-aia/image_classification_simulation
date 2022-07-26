from setuptools import setup, find_packages


setup(
    name="image_classification_simulation",
    version="0.0.1",
    packages=find_packages(
        include=[
            "image_classification_simulation",
            "image_classification_simulation.*",
        ]
    ),
    python_requires=">=3.8",
    install_requires=[
        "datasets==2.2.2",
        "easyfsl",
        "flake8",
        "flake8-docstrings",
        "gitpython",
        "jupyter",
        "mlflow==1.15.0",
        "numpy=1.22.4",
        "orion>=0.1.14",
        "pandas>=1.4.2",
        "protobuf==3.20.1",
        "pillow>=7",
        "pytorch_lightning==1.2.7",
        "pyyaml>=5.3",
        "pytest>=4.6",
        "pytest-cov",
        "recommonmark",
        "scikit-learn=1.1.1",
        "scipy=1.8.1",
        "sphinx",
        "sphinx-autoapi",
        "sphinx-rtd-theme",
        "sphinxcontrib-napoleon",
        "sphinxcontrib-katex",
        "transformers==4.19.2",
        "torchvision>=0.5.0",
        "torch==1.11.0",
        "tqdm",
    ],
    entry_points={
        "console_scripts": ["main=image_classification_simulation.main:main"],
    },
)
