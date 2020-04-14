import setuptools

import os
from io import open  # for Python 2 (identical to builtin in Python 3)

from setuptools import find_packages, setup


def pyload(name):
    ns = {}
    with open(name, encoding="utf-8") as f:
        exec(compile(f.read(), name, "exec"), ns)
    return ns


# In case it's Python 2:
try:
    execfile
except NameError:
    pass
else:
    def pyload(path):
        ns = {}
        execfile(path, ns)
        return ns


repo_root = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(repo_root, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


ns = pyload(os.path.join(repo_root,  "dpmmpython", "release.py"))
version = ns["__version__"]



setup(name='dpmmpython',
      version=version,
      description="Python wrapper for DPMMSubClusters julia package",
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Or Dinari',
      author_email='dinari@post.bgu.ac.il',
      license='MIT',
      keywords='julia python',
      classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        #'Intended Audience :: Developers',

        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
      ],
      url='https://github.com/BGU-CS-VIL/DPMMPython',
      project_urls={
          "Source": "https://github.com/BGU-CS-VIL/DPMMPython",
          "Tracker": "https://github.com/BGU-CS-VIL/issues",
          "Documentation": "https://bgu-cs-vil.github.io/DPMMSubClusters.jl/latest/",
      },
      packages=find_packages(),
      install_requires=[
          'julia','wget'
      ],
      extras_require={
          # Update `ci/test-upload/tox.ini` when "test" is changed:
          "test": [
              "numpy",
              "ipython",
              # pytest 4.4 for pytest.skip in doctest:
              # https://github.com/pytest-dev/pytest/pull/4927
              "pytest>=4.4",
              "mock",
          ],
      },
      )