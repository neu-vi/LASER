from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(
        "inference_engine/utils/*.pyx",
        compiler_directives={'language_level': "3"}
    ),
)
