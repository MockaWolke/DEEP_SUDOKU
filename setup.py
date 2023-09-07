from setuptools import setup, find_packages

setup(
    name='deepsudoku',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[ 
        'gymnasium==0.29.1',
        'matplotlib==3.7.2',
        'numpy==1.25.2',
        'pandas==2.0.3',
        'tqdm==4.66.1',
        "seaborn"
    ],
)