import argparse

def get_input_args_train():
    parse = argparse.ArgumentParser()
    
    parse.add_argument('data_dir')
    parse.add_argument('--save_dir', default='checkpoints/')
    parse.add_argument('--arch', default='vgg16')
    parse.add_argument('--learning_rate', default=0.001)
    parse.add_argument('--hidden_units', default=4096)
    parse.add_argument('--epochs', default=10)
    parse.add_argument('--gpu', default='on')
    
    return parse.parse_args()

def get_input_args_predict():
    parse = argparse.ArgumentParser()
    
    parse.add_argument('--top_k', default=3)
    parse.add_argument('--category_names', default='cat_to_name.json')
    parse.add_argument('--gpu', default='on')
    parse.add_argument('image_path', default="flowers/test/11/image_03151.jpg")
    parse.add_argument('checkpoint_path', default="checkpoints/checkpoints2.pth")
    
    return parse.parse_args()