import distutils.core
import Cython.Build
import numpy

#distutils.core.setup(ext_modules = Cython.Build.cythonize("logicops/count.pyx"), include_dirs=[numpy.get_include()])
#distutils.core.setup(ext_modules = Cython.Build.cythonize("imgops/optflow.pyx"), include_dirs=[numpy.get_include()])
distutils.core.setup(ext_modules = Cython.Build.cythonize("imgops/optflow_realsense.pyx"), include_dirs=[numpy.get_include()])