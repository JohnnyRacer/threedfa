import setuptools
from setuptools import setup, Extension
#from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="threedfa",
    version="0.0.2",
    author="JohnnyRacer",
    cmdclass={'build_ext': build_ext},
    include_package_data=True,
  #  author_email="testrr@13.com",
    description="A quick and easy AIO solution for facial detection and landmark extraction.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    #project_urls={
        #"Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    #},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
               data_files=[

        'src/threedfa/configs/BFM_UV.mat',
        "src/threedfa/Sim3DR/lib/rasterize.h"
        
        ],
        ext_modules=[         
                            Extension(name="threedfa.Sim3DR.Sim3DR_Cython",
                            sources=["src/threedfa/Sim3DR/lib/rasterize.pyx", "src/threedfa/Sim3DR/lib/rasterize_kernel.cpp"],
                            language='c++',
                            include_dirs=['src/threedfa/Sim3DR/lib',numpy.get_include()],
                            extra_compile_args=["-std=c++11"]),
                           
                            Extension(
                            name="threedfa.FaceBoxes.utils.nms.cpu_nms",
                            sources=["src/threedfa/FaceBoxes/utils/nms/cpu_nms.pyx"],
                            extra_compile_args= ["-Wno-cpp", "-Wno-unused-function"],
                            
                            include_dirs=[numpy.get_include()])
                           
                        ],
    package_dir={"": "src"},
    
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
     install_requires = ['torch', 'torchvision', 'matplotlib', 'numpy', 'opencv-python', 'imageio-ffmpeg', 'pyyaml', 'tqdm', 'argparse', 'cython', 'scikit-image', 'scipy', 'onnxruntime', 'gradio']
)
