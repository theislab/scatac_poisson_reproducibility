from setuptools import setup, find_packages

setup(
    name='poisson_atac',
    author="Laura Martens",
    author_email="laura.martens@helmholtz-muenchen.de",
    description="description",
    long_description="long_description",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        'anndata',
        'scanpy',
        'numpy==1.20.*',
        'pandas',
        'h5py==3.5.0',
        'jupyterlab',
        'pysam==0.18.0',
        'scvi-tools'
    ],
    version='1.0.0'
)
