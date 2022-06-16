from setuptools import setup, find_namespace_packages


setup(
    version="0.0.1",
    name="exact-pytorch",
    long_description="EXACT loss implementation for PyTorch.",
    url="https://github.com/tinkoff-ai/exact",
    author="Ivan Karpukhin (Tinkoff)",
    author_email="i.a.karpukhin@tinkoff.ru",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.12.0",
        "scipy>=1.5.0",
        "torch>=1.9.0"
    ]
)
