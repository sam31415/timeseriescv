from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='timeseriescv',
      version='0.1',
      description='Scikit-learn style cross-validation classes for time series data',
      long_description=readme(),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering',
      ],
      keywords='machine-learning cross-validation scikit-learn time-series',
      url='',
      author='Samuel Monnier',
      author_email='samuel.monnier@gmail.com',
      license='MIT',
      packages=['timeseriescv'],
      install_requires=[
          'numpy', 'pandas'
      ],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      )
