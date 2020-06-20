import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)         #gray image->in_channel=1,out_channels=num_filter
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(800,500)
        self.fc2 = nn.Linear(500,10)

    def forward(self, x):
        # x :  1 28 28
        x = F.relu(self.conv1(x))  # 20 (28-5+1) (28-5+1) => 20 24 24
        x = F.max_pool2d(x, 2, 2)  # 20 12 12
        x = F.relu(self.conv2(x))  # 50 (12-5+1) (12-5+1) =>50 8 8
        x = F.max_pool2d(x, 2, 2)  # 50 4 4
        x = x.view(-1, 50*4*4)  # 1 800
        x = F.relu(self.fc1(x))  # 1 500
        x = self.fc2(x)  # 1 10

        return F.log_softmax(x, dim=1)  # probability

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for idx, (data, target) in enumerate(train_dataloader):
        pred = model(data)
        loss = F.nll_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            print("Train Epoch: {}, iteration: {}, Loss: {}".format(
                epoch, idx, loss.item()))

def test(model, device, test_loader):
    model.eval()
    total_loss = 0.
    correct = 0.
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)  # batch_size * 10
            total_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)  # batch_size * 1
            correct += pred.eq(target.view_as(pred)).sum().item()

    total_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset) * 100.
    print("Test loss: {}, Accuracy: {}".format(total_loss, acc))


mnist_data = datasets.MNIST("./mnist_data",train=True,download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor()
                            ]))
# calculate mean and variance
data = [d[0].data.cpu().numpy() for d in mnist_data]
print("mean : ",np.mean(data), " variance : ", np.std(data) )  # mean :  0.13066062  variance :  0.30810776
print("shape of one image : ",data[0][0].shape)  # shape of one image :  (28, 28)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
train_dataloader = torch.utils.data.DataLoader(datasets.MNIST("./mnist_data" , train=True , download=True ,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])),
                            batch_size=batch_size,shuffle=True,
                            num_workers=1, pin_memory=True
                            )
test_dataloader = torch.utils.data.DataLoader(datasets.MNIST("./mnist_data",train=False,download=True,
                                                                      transform=transforms.Compose([
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize((0.1307,), (0.3081,))
                                                                      ])),
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=1, pin_memory=True
                                              )

lr = 0.05
momentum = 0.5
model = Net().to(device)
optimizer = optim.SGD(model.parameters() , lr = lr , momentum=momentum)

num_epoches = 2
for epoch in range(num_epoches):
    train(model, device, train_dataloader, optimizer, epoch)
    test(model, device, test_dataloader)

torch.save(model.state_dict(), "mnist_cnn.pt")
