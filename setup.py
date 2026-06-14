from setuptools import setup, find_packages
setup(
    name="aerial_catching",
    version="0.1.0",
    description="Visual-Inertial ArUco hover — sim-to-real RL pipeline",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        'gymnasium',
        'torch',
        'ruamel.yaml',
        'seaborn',
        'matplotlib',
        'warp-lang',
        'optuna',
    ],
)
