import torch
import numpy as np
def fizz_buzz_encode(i):
    if i % 15 == 0:return 3
    elif i % 5 == 0:return 2
    elif i % 3 == 0:return 1
    else:return 0

def fizz_buzz_decode(i,prediction):
    return [str(i),'fizz','buzz','fizzbuzz'][prediction]
"""
print(fizz_buzz_decode(1, fizz_buzz_encode(1)))
print(fizz_buzz_decode(2, fizz_buzz_encode(2)))
print(fizz_buzz_decode(5, fizz_buzz_encode(5)))
print(fizz_buzz_decode(12, fizz_buzz_encode(12)))
print(fizz_buzz_decode(15, fizz_buzz_encode(15)))
"""


NUM_DIGITS = 10
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

trX=torch.Tensor([binary_encode(i,NUM_DIGITS) for i in range(101,2**NUM_DIGITS)])
trY=torch.LongTensor([fizz_buzz_encode(i) for i in range(101,2**NUM_DIGITS)])
Hidden_Layer = 100
model = torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS,Hidden_Layer),
    torch.nn.ReLU(),
    torch.nn.Linear(Hidden_Layer,4)
)
critierion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.05)
Batch_size = 128
for epoch in range(15000):
    for start in range(0,len(trX),Batch_size):
        end = start+Batch_size
        X = trX[start:end]
        Y = trY[start:end]
        Y_pred = model(X)
        loss = critierion(Y_pred,Y)
        print("epoch : " , epoch ,"  loss : ",loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss = critierion(model(trX), trY).item()
    print('Epoch:', epoch, 'Loss:', loss)

test = torch.Tensor([binary_encode(i,NUM_DIGITS) for i in range(1,101)])
with torch.no_grad():
    pred = model(test)

predictions = zip(range(1, 101), list(pred.max(1)[1].data.tolist()))

print([fizz_buzz_decode(i, x) for (i, x) in predictions])