#変更点
#さらなる変更点
import time

import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models


device = 'cuda' if torch.cuda.is_available() else 'cpu'

ds_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)])

ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ds_transform
)

ds_test = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ds_transform
)

ds = 64
dataloader_train=torch.utils.data.DataLoader(
    ds_train,
    batch_size=ds,
    shuffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=ds,
    shuffle=False
)

for image_batch, label_batch in dataloader_test:
    print(image_batch.shape)
    print(label_batch.shape)
    break


model= models.MyModel()

time_strat = time.time()#???

acc_train = models.test_accuracy(model, dataloader_train)
time_end = time.time()
print(f'train accuracy:{acc_train*100:3f}%')
acc_test = models.test_accuracy(model, dataloader_test)
print(f'test accuracy:{acc_test*100:3f}{time_end-time_strat}%')

loss_fn = torch.nn.CrossEntropyLoss()

learning_rata = 1e-3
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rata)

n_epochs = 5
loss_train_history =[]
loss_test_history = []
acc_train_history = []
acc_test_history = []
for k in range(n_epochs):
    print(f'epoch{k+1}/{n_epochs}')

    time_start =time.time()
    loss_train = models.train(model, dataloader_train, loss_fn, optimizer)
    time_end = time.time()
    loss_train_history.append(loss_train)
    print(f'train loss:{loss_train:.3f}({time_end-time_start}s)', end =',')
    
    time_start =time.time()
    loss_test = models.train(model, dataloader_test, loss_fn, optimizer)
    time_end = time.time()
    loss_test_history.append(loss_test)
    print(f'train loss:{loss_test:.3f}({time_end-time_start}s)',end =',')    
    
    if (k+1)%5==0:
        time_start =time.time()
        acc_train = models.train(model, dataloader_train, loss_fn, optimizer)
        time_end = time.time()
        acc_train_history.append(acc_train)
        print(f'train accuracy:{acc_test*100:.3f}%({time_end-time_start}s)', end=',')

        time_start =time.time()
        acc_test = models.test_accuracy(model, dataloader_test)
        time_end = time.time()
        acc_test_history.append(acc_test)
        print(f'test accuracy:{acc_test*100:.2f}%({time_end-time_start}s)',end=',')


plt.plot(acc_train_history, label='train')
plt.plot(acc_test_history, label='test')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.grid()
plt.show()

plt.plot(loss_train_history, label='train')
plt.plot(loss_test_history, label='test')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.grid()
plt.show()