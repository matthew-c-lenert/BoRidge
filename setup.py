from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='boridge',
      version='0.1.10',
      description='A library of functions for selecting features using bootstrapped ridge regression',
      classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Medical Science Apps.'
      ],
      long_description=long_description,
      long_description_content_type="text/markdown",
      keywords='feature selection model evaluation',
      url='https://github.com/matthew-c-lenert/BoRidge',
      author='MC Lenert',
      author_email='matthew.c.lenert@gmail.com',
      license='MIT',
      packages=['boridge'],
      install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'joblib',
          'datetime',
      ],
      zip_safe=False)

