import numpy as np
# import matplotlib.pyplot as plt
import json
def plot_inference(img, classes, probs, model):
    
    #Label Mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f, strict=False)
    
    #   convert the cuda tensor to cpu tensor and then to a numpy array
    np_classes = classes.cpu().numpy()
    np_probs = probs[0].cpu().numpy()
    print(type(np_probs))
#   np_labels = labels.cpu().numpy()

    #create to new arrays for getting the flower name
    flower_idx = np.array([])
    flowers_names = np.array([])
#   label = ''
    
#   loop through the image desired and getting the indexes of top 5 flowers that the image could be
    for i in np_classes:
        flower_idx = np.append(flower_idx, i)

        

#   inverse the map to get the name of the flowers in cat_to_name
    inv_map = {v: k for k, v in model.class_to_idx.items()}
#   iterate over the dict to get the names
    for i in flower_idx:
        flowers_names = np.append(flowers_names, cat_to_name[inv_map[i]])
        
#   Ground truth
#   label = cat_to_name[inv_map[np_labels[num_img]]]
    
#   plot both the image of the flower and the horizontal chart
#     plt.barh(flowers_names, np_probs)
    
#     plt.title(f"The predicted name of the flower is: {flowers_names[0]}")

    for i in range(len(flowers_names)):
        print(f'Flower #{i+1}: {flowers_names[i]}, probability: {np_probs[i]*100}%')