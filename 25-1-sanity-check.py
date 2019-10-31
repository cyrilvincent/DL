import ctypes
import imp
import sys
from ctypes.util import find_library

def main():
    check()

def check():
    try:
        import tensorflow as tf
        print("TensorFlow successfully installed.")
        if tf.test.is_built_with_cuda():
            print("The installed version of TensorFlow includes GPU support.")
        else:
            print("The installed version of TensorFlow does not include GPU support.")
            #sys.exit(0)
    except ImportError as e:
        print("ERROR: Failed to import the TensorFlow module.")
        print('Reason: {}'.format(e))

    candidate_explanation = False

    python_version = sys.version_info.major, sys.version_info.minor
    print("- Python version is %d.%d." % python_version)
    if not (python_version == (3, 5) or python_version == (3, 6)):
        candidate_explanation = True
        print("- The official distribution of TensorFlow for Windows requires "
          "Python version 3.5 or 3.6.")

    try:
        _, pathname, _ = imp.find_module("tensorflow")
        print("- TensorFlow is installed at: %s" % pathname)
    except ImportError:
        candidate_explanation = False
        print("""- No module named TensorFlow is installed in this Python environment. You may
install it using the command `pip install tensorflow`.""")

    try:
        msvcp140 = ctypes.WinDLL("msvcp140.dll")
        msvcp140_path = find_library("msvcp140.dll")
        print('- msvcp140.dll Found at {}'.format(msvcp140_path))
    except OSError:
        candidate_explanation = True
        print("""
- Could not load 'msvcp140.dll'. You may install this DLL by downloading Microsoft Visual
  C++ 2015 Redistributable Update 3 from this URL:
  https://www.microsoft.com/en-us/download/details.aspx?id=53587""")

    try:
        cudart64_90 = ctypes.WinDLL("cudart64_90.dll")
        cudart64_90_path = find_library("cudart64_90.dll")
        print('- Cuda 9.0 found at {}'.format(cudart64_90_path))
        # TODO: Look for version.txt file in CUDA path
    except OSError:
        candidate_explanation = True
        print("""
- Could not load 'cudart64_90.dll'. Download and install CUDA 9.0 from
  this URL: https://developer.nvidia.com/cuda-toolkit""")

    try:
        nvcuda = ctypes.WinDLL("nvcuda.dll")
        nvcuda_path = find_library("nvcuda.dll")
        print('- nvcuda.dll found at {}'.format(nvcuda_path))
    except OSError:
        candidate_explanation = True
        print("""
- Could not load 'nvcuda.dll'. The GPU version of TensorFlow requires that
  this DLL be installed in a directory that is named in your %PATH%
  environment variable. Typically it is installed in 'C:\Windows\System32'.
  If it is not present, ensure that you have a CUDA-capable GPU with the
  correct driver installed.""")

    cudnn7_found = False
    try:
        cudnn7 = ctypes.WinDLL("cudnn64_7.dll")
        cudnn7_path = find_library("cudnn64_7.dll")
        print("- cuDNN Found at {}".format(cudnn7_path))
        cudnn7_found = True
    except OSError:
        candidate_explanation = True
        print("""
- Could not load 'cudnn64_7.dll'. The GPU version of TensorFlow
  requires that this DLL be installed in a directory that is named in
  your %PATH% environment variable. Note that installing cuDNN is a
  separate step from installing CUDA, and it is often found in a
  different directory from the CUDA DLLs. You may install the
  necessary DLL by downloading cuDNN 7.0 for Cuda 9.0 from this URL:
  https://developer.nvidia.com/cudnn""")

    if not candidate_explanation:
        print("""
- All required DLLs appear to be present. Please open an issue on the
  TensorFlow GitHub page: https://github.com/tensorflow/tensorflow/issues""")

    sys.exit(-1)


if __name__ == '__main__':
    main()