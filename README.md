# Image_Classifier
This project uses convolutional neural network to train an image classifier that is able to identify 102 different flower species with 94% testing accuracy. This image classifier can be used to identify flower species from new images, e.g., in a phone app that tells you the name of the flower your camera is looking at.

## 1. Problem to solve
Build an application that tells the name of a flower from an image as input.

## 2. Available data
[102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) was given by the AI Programming with Python Nanodegree program. This dataset contains 102 flower species with their respective labels. This images have different sizes.

Data file structure:

- `flowers`: folder that contains image data.
    - `train`, `valid`, `test`: subfolders for training, validating and testing the image classifier respectively.
        - `1`, `2`, ..., `102`: 102 subfolders. The subfolders name correspond to the flower categories. Given the large data size, data folders are not included here.
  
## 3. What I did 

[Main Code](link)

1. Data loading and data preprocessing
    - Load image data
    - Training set: apply transformations such as rotation, scaling, and horizontal flipping (model generalizes / performs better)
    - All datasets: Resize and crop to the appropriate image size (required by pre-trained model)
    - All datasets: Normalize image colors (RGB) using mean and standard deviation of pre-trained model
    - Training set: data shuffled at each epoch
   
2. Build and train the model
    - Load a pre-trained network `VGG16` ([reference](https://arxiv.org/pdf/1409.1556v6.pdf)) and freeze parameters
    - Define a new, untrained neural network, as a new classifier. The classifier has a hidden layer (ReLU activation) and an output layer (LogSoftmax activation). In this classifier was assigned dropout to deal with overfitting.
    - Assign criterion (NLLLoss, Negative Nog Loss) and optimizer (Adam, Adaptive Moment Estimation, [reference](https://arxiv.org/abs/1412.6980))
    - Train the classifier layers using forward and backpropagation on GPU
    - Track the loss and accuracy on the validation set to determine the best hyperparameters

3. Use the trained classifier to predict image content

    - Test trained model on testing set (94% accuracy)
    - Save trained model as checkpoint
    - Write a function that preprocess images (resizes keeping the aspect ratio, crops, normalizes color channels, and normalizes again with the respective values of the mean and the standard deviation for each color channel)
    - Write a function that gives top-5 most probable flower names based on an image

4. Build a command line application

    - See below for details

<img src="assets/inference_example.png" width=300>

## 4. How to run the command line application

- #### Train the image classifier

    [`train.py`](train.py): Train the image classifier, report validation accuracy along training, and save the trained model as a checkpoint.

    - Basic usage:
        - Specify directory of image data: `python train.py flowers`

    - Options:
        - Set directory to save checkpoints: `python train.py flowers --save_dir 'checkpoints/'.`
        - Choose architecture: `python train.py flowers --arch "vgg16".`
        - Set hyperparameters: `python train.py flowers --learning_rate 0.001 --hidden_units 512 --epochs 10.`
        - Use GPU for training: `python train.py flowers --gpu 'on'.`

- #### Identify flower name from a new flower image

    [`predict.py`](predict.py): Use the trained image classifier to predict flower name along with the probability of that name

    - Basic usage: 
        - Specify file path of the image and directory name of saved checkpoint: `python predict.py 'flowers/test/1/image_06743.jpg' 'checkpoints/'`

    - Options:
        - Return top K most likely classes: `python predict.py 'flowers/test/11/image_03151.jpg' 'checkpoints/' --top_k 3.`
        - Use a mapping of categories to real names: `python predict.py 'flowers/test/11/image_03151.jpg' 'checkpoints/' --category_names 'cat_to_name.json'.`
        - Use GPU for inference: `python predict.py 'flowers/test/11/image_03151.jpg' 'checkpoints/' --gpu 'on'.`
