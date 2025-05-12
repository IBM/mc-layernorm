from setuptools import setup, find_packages

setup(
    name='mcln',
    url='https://github.com/IBM/mc-layernorm',
    author='Thomas Frick',
    author_email='fri@zurich.ibm.com',
    # Needed to actually package something
    packages=find_packages(),
    # Needed for dependencies
    install_requires=['torch'],
    # *strongly* suggested for sharing
    version='1.0',
    # The license can be anything you like
    license='Appache-2.0',
    description='MonteCarlo Layer Normalization Pytorch Implementation',
)
