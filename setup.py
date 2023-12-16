import numpy
from setuptools import setup, find_packages

setup(
    name='YourProjectName',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',  # List other dependencies here
        'pandas',
        'scikit-learn',
        'data',
        'warnings',
        'math',
        'matlablib',
    ],
)
    # Additional metadata about your project
#    author='Your Name',
#    author_email='your.email@example.com',
#    description='A brief description of your project',
#    url='http://yourprojecthomepage.com',
#)
