# Image_Classifier
This project uses convolutional neural network to train an image classifier that is able to identify 102 different flower species with 94% testing accuracy. This image classifier can be used to identify flower species from new images, e.g., in a phone app that tells you the name of the flower your camera is looking at.

# 1. Problem to solve
Build an application that tells the name of a flower from an image as input.

# 2. Available data
[102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) was given by the AI Programming with Python Nanodegree program. This dataset contains 102 flower species with their respective labels. This images have different sizes.

Data file structure:

- `flowers`: folder that contains image data.
    - `train`, `valid`, `test`: subfolders for training, validating and testing the image classifier respectively.
        - `1`, `2`, ..., `102`: 102 subfolders. The subfolders name correspond to the flower categories. Given the large data size, data folders are not included here.
  
## 3. What I did 
