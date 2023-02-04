# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
import torch
from torch import nn, optim
from torchvision import models, transforms, datasets
import torch.nn.functional as F

import numpy as np
# import matplotlib.pyplot as plt

from collections import OrderedDict

from get_input_args import get_input_args_train

def main():

    in_arg = get_input_args_train()


    #Load the Data
    data_dir = in_arg.data_dir

    print(data_dir)
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    valid_dir = data_dir + '/valid'

    #Define Transforms
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229 ,0.224 ,0.225])])

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229 ,0.224 ,0.225])])

    #Load Datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transforms)

    #Define dataloaders using the datasets and transforms
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)



    #training
    if in_arg.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        #Freeze parameters
        for param in model.parameters():
            param.requires_grad  = False

        model.classifier = nn.Sequential(nn.Linear(25088, int(in_arg.hidden_units)),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(int(in_arg.hidden_units), 102),
                                        nn.LogSoftmax(dim=1))

    elif in_arg.arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        #Freeze parameters
        for param in model.parameters():
            param.requires_grad  = False

        model.classifier = nn.Sequential(nn.Linear(9216, int(in_arg.hidden_units)),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(int(in_arg.hidden_units), 102),
                                         nn.LogSoftmax(dim=1))


    device = torch.device("cuda:0" if torch.cuda.is_available() and in_arg.gpu == "on" else "cpu")


    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(in_arg.learning_rate))
    model.to(device)
    epochs = int(in_arg.epochs)
    model.to(device)

    valid_losses, train_losses = [],[]

    for e in range(epochs):
        running_loss = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            logits = model.forward(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            valid_loss = 0
            accuracy = 0

            with torch.no_grad():
                model.eval()
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    ps = torch.exp(logits)
                    valid_loss += criterion(logits, labels)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            train_losses.append(running_loss/len(trainloader))
            valid_losses.append(valid_loss/len(validloader))

            print(f'Epoch: {e+1}',
                  f'Accuracy: {accuracy.item()*100}%',
                 "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                 "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                 "Valid Accuracy: {:.3f}.. ".format(accuracy/len(validloader)))

            model.train()



    print('Done!!!')

     #Save Checkpoint
    model.class_to_idx = train_dataset.class_to_idx

    checkpoint = {'output_size': 102,
                  'model':model,
                  'state_dict': model.state_dict(),
                  'epochs': epochs,
                  'criterion': criterion,
                  'optimizer_state_dict': optimizer.state_dict,
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, in_arg.save_dir+'checkpoints2.pth')

    
# Call to main function to run the program
if __name__ == "__main__":
    main()

