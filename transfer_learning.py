import numpy as np
import torchvision
import torch
import torch.nn as nn
from torchvision import models , transforms , datasets

import time
import os
import copy

data_dir = "./hymenoptera_data"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"
# Number of classes in the dataset
num_classes = 2
# Batch size for training (change depending on how much memory you have)
batch_size = 32
# Number of epochs to train for
num_epochs = 15
# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True
# size of the input image
input_size = 224

# transform image
data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

    ]),
    "val":transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

    ])
}
# input images
img_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transform[x]) for x in ["train", "val"]}

# data loader
# folder train and val respectively includes a train and val folder, so label can be extracted automatically by dataloader
dataloaders_dict = {x: torch.utils.data.DataLoader(img_datasets[x],batch_size=batch_size, shuffle=True, num_workers=4) for x in ["train", "val"]}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_parameter_requires_grad(model_ft, feature_extract):
    if feature_extract:
        for param in model_ft.parameters():
            param.requires_grad = False


# download model, set all params to not need grad, only make fc to be true
def initialize_model(model_name , num_classes , feature_extract,use_pretrained=True):
    if model_name == "resnet":
        model = models.resnet18(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        print("model not implemented")
        return None, None
    return model, input_size


model_ft, input_size = initialize_model(model_name,
                    num_classes, feature_extract, use_pretrained=True)
"""
print(model_ft)
print(model_ft.layer1[0].conv1.weight.requires_grad)
print(model_ft.fc.weight.requires_grad)
"""



def train_model(model, dataloaders, loss_fn, optimizer, num_epochs=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.
    val_acc_history = []
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            running_loss = 0.
            running_corrects = 0.
            if phase == "train":
                model.train()
            else:
                model.eval()
            count = 0.
            # train folder has 244 images(ants and bees), batch_size is 32, so for loop runs ceil(244/32) = 8 times
            # val folder has 153 images(ants and bees), batch_size is 32, so for loop runs ceil(153/32) = 5 times

            for inputs, labels in dataloaders[phase]:
                count += 1.
                inputs, labels = inputs.to(device), labels.to(device)

                with torch.autograd.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)  # bsize * 2
                    loss = loss_fn(outputs, labels)  # loss of a batch, scalar

                preds = outputs.argmax(dim=1)
                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * inputs.size(0)  # inputs.size(0):batch_size; ruuning_loss sum up the loss of each batch
                running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()  # num of correct samples in a batch;

            epoch_loss = running_loss / len(dataloaders[phase].dataset)# loss of all train(val) images / num of train(val) images = average loss for one image
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            print("Phase {} loss: {}, acc: {}".format(phase, epoch_loss, epoch_acc))

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc)
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


model_ft = model_ft.to(device)
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                   model_ft.parameters()), lr=0.001, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()
_, ohist = train_model(model_ft, dataloaders_dict, loss_fn, optimizer, num_epochs=num_epochs)
