from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(
        "inference_engine/utils/*.pyx",
        compiler_directives={'language_level': "3"},
        force=True
    ),
    include_dirs=[np.get_include()]
)
