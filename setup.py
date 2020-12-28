from setuptools import setup, find_packages

setup(
    name='gin-sweep',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/Vatican-X-Formers/gin-sweep',
    license='MIT License',
    author='syzymon',
    author_email='',
    description='Creating gin configs out of .yaml sweep files',
    python_requires='>=3.6',
    install_requires=[
        'gin-config >= 0.4.0'
        'pyyaml>=5.3.1',
        'pathvalidate'
    ]
)
