from setuptools import setup, find_packages
setup(
    name="visual_quadrotor_control",
    version="0.1.0",
    description="Visual representation learning and quadrotor control",
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
