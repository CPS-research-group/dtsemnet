from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("cro_dt.cythonfns.TreeEvaluation",           # Module name
              sources=["cro_dt/cythonfns/TreeEvaluation.pyx"],
              include_dirs=[np.get_include()])
]

setup(
    ext_modules = cythonize(extensions,
                            compiler_directives={'language_level': "3"},
                            annotate=True)
)
