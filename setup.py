"""
Copyright (C) 2019 Quentin Peter

This file is part of WF_NTP.

WF_NTP is distributed under CC BY-NC-SA version 4.0. You should have
recieved a copy of the licence along with WF_NTP. If not, see
https://creativecommons.org/licenses/by-nc-sa/4.0/.
"""
from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='WF_NTP',
      version='3.3.3',
      description='Wide-field nematode tracking platform.',
      long_description=readme(),
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'License :: Free for non-commercial use',
          'Programming Language :: Python :: 3 :: Only',
          'Topic :: Scientific/Engineering',
      ],
      keywords='nematode worm tracker',
      url='https://github.com/impact27/WF_NTP',
      author='TODO',
      author_email='TODO',
      license='CC BY-NC-SA 4.0',
      packages=find_packages(),
      install_requires=[
          "numpy>=1.10.0",
          "mahotas>=1.4.0",
          "matplotlib>=1.4.3",
          "opencv_python>=2.4.12",
          "pandas>=0.16.2",
          "Pillow>=2.9.0",
          "PIMS>=0.2.2",
          "scikit_image<0.16",
          "scipy>=0.16.0",
          "tifffile>=2015.8.17",
          "trackpy>=0.2.4"
      ],
      scripts=[
          'run_script/multiwormtracker_app.py',
          'run_script/multiwormtracker_app',
      ],
      # test_suite='nose.collector',
      # tests_require=['nose', 'nose-cover3'],
      include_package_data=True,
      zip_safe=False)
