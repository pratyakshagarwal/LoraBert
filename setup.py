from setuptools import setup, find_packages

setup(
    name='LoraBert',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        # Add your project's dependencies here
        # For example: 'numpy', 'torch'
    ],
    include_package_data=True,
    description='This lib will contain helper functions for AI use',
    author='pratyakshagarwal',
    author_email='pratyakshagarwal93@gmail.com',
    license='Apache 2.0',
)
# pip install -e .