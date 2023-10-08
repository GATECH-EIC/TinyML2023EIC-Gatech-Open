​
​
### 1. For Deployment
​
**All settings are ready in the project file.  If the project file can be opened normally, it does not need to be modified.**
​
Please open the *Deployment/MDK-ARM/deploy2.uvprojx* in Keil6. While ensuring proper hardware connectivity, build and download the project. 
​
​
​
### 2. Details of the setting for your understanding
​
In case you want to replicate our work，please follow the setting below: 
​
​
​
For CubeMX Settings
​
```
1. Project Manager -> Advanced Settings: set "RCC","GPIO","CRC" in LL, keep "USART" in HAL.
2. Pinout -> System Core -> NVIC -> Code generation: only keep "Time base: System tick timer"
```
​
For Keil Compiling Settings
​
```
1.Option for Target->Target->Code generation：Use default compiler version 6 & Use Micro LIB;
2.Option for Target->C/C++(AC6)->Optimization:-Oz
3.Option for Target->C/C++(AC6)->Click on "One ELF Section per Function" & "Execute-only Code" & "Short enums/wchar"
```
​
​
​
**Note:**  We did not use X-CUBE-AI due to its large memory overhead.
​
In addition to the necessary/default files for the STM32 system, we also have several files that need to be included during compilation.
​
```
matrix.c / .h
aiRun.c / .h
parameter.c / .h
```
​
Our method extracts some features and inputs them into a three-layer FC.
​
- *matrix.c / .h*:  Defines functions for matrix data format, calculating matrix multiplication, matrix addition, and Relu.
- *parameter.c / .h*: Stores FC parameters and hyperparameters when calculating features.
- *aiRun.c / .h* : 
  - *Model_Init()*: We deconvert our homemade float16 data to float32 and convert them into matrix data.
  - *aiRun()*: We do feature extraction and matrix calculation here.
​
​
​
*weight_genenerate/to_float16.c* file should be compiled and run on PC. This file manually sets the model's original float weights and defines a set of structures that compress two float16 data into a uint32 for the weights in *parameter.c*.
​
