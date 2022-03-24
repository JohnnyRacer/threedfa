'''
python setup.py build_ext -i
to compile
'''

from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy


setup(
    name='Sim3DR_Cython',  # not the package name
    cmdclass={'build_ext': build_ext},
    data_files=['lib/rasterize.h'],
    ext_modules=[Extension("Sim3DR_Cython",
                           sources=["lib/rasterize.pyx", "lib/rasterize_kernel.cpp"],
                           language='c++',
                           include_dirs=['lib',numpy.get_include()],
                           extra_compile_args=["-std=c++11"])],
)
