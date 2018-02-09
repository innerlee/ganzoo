import sys
from setuptools import find_packages, setup

install_requires = ['numpy>=1.11.1', 'torch', 'torchvision']
if sys.version_info < (3, 3):
    install_requires.append('backports.shutil_get_terminal_size')
if sys.version_info < (3, 4):
    install_requires.append('enum34')


def readme():
    with open('README.md') as f:
        content = f.read()
    return content


def get_version():
    version_file = 'ganzoo/version.py'
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec')) # pylint: disable=W0122
    return locals()['__version__']


setup(
    name='ganzoo',
    version=get_version(),
    description='Implementation of all GANs',
    long_description=readme(),
    keywords='gan',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Utilities',
    ],
    url='https://github.com/innerlee/ganzoo',
    author='Zhizhong Li',
    author_email='363664470@qq.com',
    license='MIT',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=install_requires,
    zip_safe=False
)  # yapf: disable
