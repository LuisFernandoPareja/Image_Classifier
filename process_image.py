from PIL import Image
import numpy as np

def process_image(image):
    
    #Aspect ratio shortest side 256
    w, h = image.size 
    aspect_ratio = w/h
    if aspect_ratio > 1:
        image = image.resize((round(aspect_ratio * 256), 256))
    else:
        image = image.resize((256, round(256 / aspect_ratio)))
        
    #Crop 224x224
    w2, h2 = image.size
    left = (w2 - 224) / 2
    top = (h2 - 224) / 2
    right =  (w2 + 224) / 2
    bottom =  (h2 + 224) / 2
    
    image = image.crop((round(left),round(top),round(right),round(bottom)))
    
    np_image = np.array(image)
    np_image = np_image / 255
    
    std = np.array([0.229, 0.224, 0.255])
    mean = np.array([0.485, 0.456, 0.406])
    np_image = ((np_image - mean) / std)
    
    
    np_image_T = np.transpose(np_image, (2,0,1))

    
    np_image_T = np.expand_dims(np_image_T, axis=0)
    
    return np_image_T