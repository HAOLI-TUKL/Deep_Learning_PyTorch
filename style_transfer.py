
from __future__ import division
from torchvision import models
from torchvision import transforms
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(image_path, transform=None, max_size=None, shape=None):
    image = Image.open(image_path)
    if max_size:
        scale = max_size / max(image.size)
        size= np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.ANTIALIAS)

    if shape:
        image = image.resize(shape, Image.LANCZOS)

    if transform:
        image = transform(image).unsqueeze(0)

    return image.to(device)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])



content = load_image("./images/content.jpg", transform, max_size=400)
style = load_image("./images/style.jpg", transform, shape=[content.size(2), content.size(3)])


"""
print(content.shape)
print(style.shape)
mean_content = [np.mean(content[0][i].data.numpy()) for i in range(3)]
std_content = [np.std(content[0][i].data.numpy()) for i in range(3)]
mean_style = [np.mean(style[0][i].data.numpy()) for i in range(3)]
std_style = [np.std(style[0][i].data.numpy()) for i in range(3)]
print(mean_content) # [0.48955464, 0.59678465, 0.63614184]
print(std_content) # [0.26077, 0.22831875, 0.23347282]
print(mean_style) # [0.3404517, 0.44545957, 0.49017707]
print(std_style) # [0.2667993, 0.2628932, 0.22826852]
"""


unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(1) # pause a bit so that plots are updated
# plt.figure()
# imshow(style[0], title='Image')
# imshow(content[0], title='Image')

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features


target = content.clone().requires_grad_(True)
optimizer = torch.optim.Adam([target], lr=0.003, betas=[0.5, 0.999])
vgg = VGGNet().to(device).eval()

target_features = vgg(target)

total_step = 150
style_weight = 100.
for step in range(total_step):
    target_features = vgg(target)
    content_features = vgg(content)
    style_features = vgg(style)

    style_loss = 0
    content_loss = 0
    for f1, f2, f3 in zip(target_features, content_features, style_features):
        content_loss += torch.mean((f1-f2)**2)
        _, c, h, w = f1.size()
        f1 = f1.view(c, h*w)
        f3 = f3.view(c, h*w)

        # 计算gram matrix
        f1 = torch.mm(f1, f1.t())
        f3 = torch.mm(f3, f3.t())
        style_loss += torch.mean((f1-f3)**2)/(c*h*w)

    loss = content_loss + style_weight * style_loss

    # 更新target
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print("Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}"
             .format(step, total_step, content_loss.item(), style_loss.item()))


denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
img = target.clone().squeeze()
img = denorm(img).clamp_(0, 1)
plt.figure()
imshow(img, title='Target Image')
plt.savefig('./images/generated.jpg')
