import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eda-and-beyond", 
    version="0.0.1",
    author="Freda Xin",
    author_email="freda.xin@gmail.com",
    description="A package of automation tools for EDA and modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FredaXin/eda_and_beyond",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
          'numpy',
          'pandas',
          'matplotlib',
          'seaborn',
          'sklearn'
      ],
    python_requires='>=3.7',

    test_suite='nose.collector',
    tests_require=['nose']
)