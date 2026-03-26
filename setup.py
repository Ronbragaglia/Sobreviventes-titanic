"""
Setup do Projeto Titanic
"""

from setuptools import setup, find_packages
from pathlib import Path

# Lê o README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="titanic-survival-prediction",
    version="2.0.0",
    author="Ron Bragaglia",
    author_email="ronbragaglia@gmail.com",
    description="Projeto de Machine Learning para prever a sobrevivência dos passageiros do Titanic",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ronbragaglia/Sobreviventes-titanic",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "plotly>=5.14.0",
        "openpyxl>=3.1.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "titanic-predictor=titanic.predictor:main",
        ],
    },
    keywords="titanic machine-learning random-forest classification survival-prediction",
    project_urls={
        "Bug Reports": "https://github.com/Ronbragaglia/Sobreviventes-titanic/issues",
        "Source": "https://github.com/Ronbragaglia/Sobreviventes-titanic",
        "Documentation": "https://github.com/Ronbragaglia/Sobreviventes-titanic#readme",
    },
)
