from setuptools import setup, find_packages

setup(
    name="water-potability-detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "pandas",
        "fastapi",
        "uvicorn",
    ],
)
