from setuptools import setup, find_packages

# Read requirements from the requirements.txt file
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='vertibench',
    version='0.1',
    packages=find_packages(),
    description='A tool for benchmarking vertical federated learning algorithms, containing synthetic data split and'
                'data evaluation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Zhaomin Wu, Junyi Hou, Bingsheng He',
    author_email='zhaomin@u.nus.edu',
    url='https://github.com/JerryLife/VertiBench',
    install_requires=required,
    classifiers=[
        # Choose from: https://pypi.org/classifiers/
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache License 2.0',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        # etc.
    ],
    # Optional
    keywords='vertical federated learning, benchmark, synthetic data, feature split, data valuation, federated learning',
    python_requires='>=3.8',
    package_data={},
    # Entry points for creating executable scripts or plugins
    entry_points={
        'console_scripts': [
            # temporarily empty, add vertical_split in the future
        ],
    },
)
