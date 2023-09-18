from setuptools import setup, find_packages

setup(
    name="poisson_atac",
    author="Laura Martens",
    author_email="laura.martens@helmholtz-muenchen.de",
    description="description",
    long_description="long_description",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "anndata",
        "scanpy",
        "jupyterlab",
        "scvi-tools",
        "seml",
        "pyranges",
        "scvelo",
        "statannotations",
        "scib",
    ],
    version="1.0.0",
)
