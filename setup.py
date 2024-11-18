from setuptools import setup, find_packages

setup(
    name="cnni",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "matplotlib>=3.4.2",
        "seaborn>=0.11.1"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Contextual Nearest Neighbor Imputation (CNNI) - A context-aware method for missing data imputation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kchemorion/cnni",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'cnni=cnni.cli:main',
        ],
    },
)
