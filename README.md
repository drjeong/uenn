# Uncertainty Computation using Evidential Deep Learning (aka ENN: Evidential Neural Network)

In the project, three workspaces are created: "UENN_testbed," "UENNLib," and "UENNLib_test." 

UENN_testbed is a stand-alone workspace that runs an ENN that computes various uncertainty computations.

UENNLib creates a DLL (dynamic linking library) to run uncertainty computation with ENN. 

UENNLib_test is a simple program that tests the created library.

## Implementation Environment

All source codes are created and tested with Visual Studio 2019 on Windows 11 OS.

It uses C++17 and Windows APIs.


## Required Libraries

* Eigen - C++ template library for linear algebra

> Eigen 3.4.0 (https://eigen.tuxfamily.org/)

> The location to download the library - https://gitlab.com/libeigen/eigen/-/releases/3.4.0

> Once downloaded, place the library under the "Lib" folder as,
```
$(SolutionDir)Lib\eigen-3.4.0
```


* Pytorch C++ CUDA


> The library was tested with libtorch-win-shared-with-deps-2.0.1+cu118.

> The location to download the library - https://pytorch.org/get-started/locally/

> Please make sure to select the preference options - LibTorch and C++ / Java.

> Once downloaded, place the library under the "Lib" folder as:

```
$(SolutionDir)Lib\libtorch-win-shared-with-deps-2.0.1+cu118

$(SolutionDir)Lib\libtorch-win-shared-with-deps-debug-2.0.1+cu118
```

> To run the program, you must copy DLLs. 


> Release Mode

```
xcopy $(SolutionDir)Lib\libtorch-win-shared-with-deps-2.0.1+cu118\libtorch\lib\*.dll $(SolutionDir)$(Platform)\$(Configuration)\ /c /y
```

> Debug Mode

```
 xcopy $(SolutionDir)Lib\libtorch-win-shared-with-deps-debug-2.0.1+cu118\libtorch\lib\*.dll $(SolutionDir)$(Platform)\$(Configuration)\ /c /y
```

## How to use the library

UENNLib creates UENNLib.dll and UENNLib.lib. When using the library, you must place both files into the same folder, where an executable file is located. 

Since the library references Pytorch C++ CUDA library, all necessary DLLs must be placed in the folder as well. 



