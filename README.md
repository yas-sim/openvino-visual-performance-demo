# OpenVINO Visual Performance Demo Program

## Overview
This is an OpenVINO performance demo program. This program demonstrates inferencing performance visually on the screen so that the audience can get the performance of the system and OpenVINO intuitively.  
The proram supports simple image classification models models.   
**NOTICE**  
The program has been updated for performance up. The previous version has performance bottle neck caused by OpenCV. The new version replaced some OpenCV portion with OpenGL. Also, some features such as SSD model support is dropped to simplify the program.  

### You can use this program to demonstrate side-by-side performance comparison demo of OpenVINO.  
![video](./resources/visual-demo.gif)

## Prerequisites
- OpenVINO 2020.4
- OpenGL, GLUT (FreeGLUT)  
`(Ubuntu) sudo apt install python-opengl freeglut3 freeglut3-dev`   
- Some Python modules  
` pip install pyyaml,numpy,opencv-python,PyOpenGL,PyOpenGL_accelerate`
- Image files for inferencing  
  - Annotation file is not required.
  - Place image files in a directory and specify the directory in the YAML configration file.
- Image classification IR model such as ResNet-50  
**NOTICE**  
- The program requires GLUT. If you are using Windows, maybe you need to install GLUT separately to enable it.  

## Configuration
Benchmark configuration can be defined in a YAML file. Create your own configuration by reffering to the `default.yml` as an example.  

## How to run
1. Create or modify YAML configuration file
 - The YAML configuration file contains the parameters for performance demo
 - `default.yml` will be used when no configuration file is given.
2. Run the performance demo script  
`python visual-demo.py -c <config.yml>`

## Tested environment
- OpenVINO 2020.4, 2021.1
- Windows 10
- Ubuntu 18.04

## Update history
 - 07-10-2020: Separated benchmark main loop to a thread. Screen update and control logic is still remain in the main thread. This minimizes the performance impact from screen update.  
 - 16-10-2020: Replaced OpenCV codes with OpenGL. Droped support for SSD models.   
