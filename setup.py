""" Setup script for taxi chargin simulator. """

from setuptools import setup


setup(
    name='simulasi-taksi',
    version='0.1.0',
    author='m-yuhas',
    author_email='m-yuhas@qq.com',
    maintainer='m-yuhas',
    url='https://github.com/m-yuhas/simulasi-pengisian-taksi-listrk',
    description='Electric taxi fleet simulator',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Environment :: GPU',
        'Environment :: MacOS X',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ],
    packages=['simulator', 'scheduler', 'scripts', 'analysis'],
    include_package_data=False,
    install_requires=[
        'coloredlogs',
        'gymnasium',
        'matplotlib',
        'numpy',
        'pandas',
        'PyYAML',
        'scikit-learn',
        'stable_baselines3',
        'torch',
    ]
)
