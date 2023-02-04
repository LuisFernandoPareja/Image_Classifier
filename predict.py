import torch
from load_checkpoint import load_checkpoint
from process_image import process_image
from imshow import imshow
from get_input_args import get_input_args_predict
from PIL import Image
from plot_inference import plot_inference

def main():
    in_arg = get_input_args_predict()

    device = torch.device("cuda:0" if torch.cuda.is_available() and in_arg.gpu == 'on' else "cpu")

    model = load_checkpoint(in_arg.checkpoint_path)

    model.to(device)

    image = Image.open(in_arg.image_path)

    image = process_image(image)

    image = torch.Tensor(image)

    image = image.to(device)


    with torch.no_grad():
        model.eval()
        output = model.forward(image)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(int(in_arg.top_k), dim=1)

    model.train()

    plot_inference(image, top_class, top_p, model)
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
