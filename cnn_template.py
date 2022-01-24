
#importing the required libraries
import torch
import numpy as np
import pandas as pd
import torchvision # contains datasets and archeteches 
import torchvision.transforms as transforms# transfermations in the torch vision 
import torch.nn as nn 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

# initiating the GPU. You need to remeber that tensor operation can be done on tensors in the same device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# In geneal before we upload the data the transfermations should be specified. However when transformations are being specified its better to normalise the data. Mean,std can be normally found in the internet.However we will not do it

"""Hi Doctor -
This is the section I dont understand,When i put the transforers seperatly i cant move forward because iter function does not work in the training set. But you can check the others section where the code works.Only difference is that the trasformer is given inside, not outside 
"""

transform = transforms.Compose([transforms.ToTensor()])

train_set=torchvision.datasets.FashionMNIST(
    root="./data/FashionMNIST",
    train=True,
    download=True,
    transform=transform
    
)

sample=next(iter(train_set))
image, label=sample
print(image.shape)#we can see it is 1 ,28 ,28 .One image by 28 28
plt.imshow(image.squeeze(),cmap='gray')# squeezing reduces one dimensions.bacically takes of the dimension with "1"
print("label",label)



"""When i put the code in the below way it works well """

#loading the data to "train_set" folder in torch. This is just a container to hold the data.
#The data then needs to be loaded to the data loader.This is done because all data cant be uploaded at once. They should be uploaded in batches.
train_set=torchvision.datasets.FashionMNIST(
    root="./data/FashionMNIST",
    train=True,
    download=True,
    transform=transforms.Compose([
                                  transforms.ToTensor()
    ])
    
)

test_set=torchvision.datasets.FashionMNIST(
    root="./data/FashionMNIST",
    train=False,
    download=True,
    transform=transforms.Compose([
                                  transforms.ToTensor()
    ])
    
)

# loader=torch.utils.data.DataLoader(train_set,batch_size=len(train_set),num_workers=1)#loading the data to the loader 
# make sure you put "batch_size=len(trainset)".This will make all traning data to one batch 
# data=next(iter(loader))
# data[0].mean(),data[0].std()#
# mean,std=data[0].mean(),data[0].std()

# Now would visualise the images to ensure it has been loaded correcly
# batch size is kept 20 , a low value to visualising purpose only
train_loader=DataLoader(train_set, batch_size=20)
test_loader=DataLoader(test_set, batch_size=20)

print(f'sample in train set : {len(train_set)}, samples in test set {len(test_set)}')

train_set.train_labels

#observing how many samples per category
train_set.train_labels.bincount()

# Visualising just one sample
sample=next(iter(train_set))
image, label=sample
print(image.shape)#we can see it is 1 ,28 ,28 .One image by 28 28
plt.imshow(image.squeeze(),cmap='gray')# squeezing reduces one dimensions.bacically takes of the dimension with "1"
print("label",label)

# we can also do the same thing as 
index=5000
image, label = train_set[index] # x is now a torch.Tensor
print(f' size of the image{image.shape}')
plt.imshow(image.numpy()[0], cmap='gray')

#Visualising in a batch 
batch =next(iter(train_loader))#previously we iterated on the train container. Train container will provide one image at a time when "next" is called.
# in the train_loader , batches are stored.Therefore when "next" is called the next batch will be given 
print(len(batch))
print(type(batch))
images, labels= batch

images.shape# we can see now that the images have 20 .

labels.shape# there are 20 labels

grid=torchvision.utils.make_grid(images,nrow=5)
plt.figure(figsize=(10,10))
plt.imshow(np.transpose(grid,(1,2,0)))

# an interseting question i had was why use non liniear activation functions , and why not use just the neural network with the linear layers . the reason is that , if we use linear functions only as an input to the other layer, it just becomes another linear function , which limits is performace. Function of another linear function is another linear function.this is the reason why we have a non linear activation functions to it 
# another question i had was , why use relu, why not sigmoid or tanh.the reson is that sigmoid and tanh , the points in sigmoid will have gradients between 0-0.25 and in tanh it will be less than one. there for , during backpropergation u will end up in a vanishing gradient problem. to avoid this relu is used. relu always has a deravative of 0 or 1. how ever this can lead to dieing neuron problem. this can be solved with leaky rely

class Network(nn.Module):
  def __init__(self):
    super(Network,self).__init__()
    self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
    self.conv2=nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5)

    self.fc1=nn.Linear(in_features=12*4*4,out_features=120)
    self.fc2=nn.Linear(in_features=120,out_features=60)
    self.out=nn.Linear(in_features=60,out_features=10)

  def forward(self,t):
      # first input layer 
      t=t

      # hidden conv layer 
      t=self.conv1(t)
      t=F.relu(t)
      t=F.max_pool2d(t,2,stride=2)

      # hidden conv layer 
      t=self.conv2(t)
      t=F.relu(t)
      t=F.max_pool2d(t,2,stride=2)
      
      # hidden linear layer 
      t=t.reshape(-1,12*4*4)
      t=self.fc1(t)
      t=F.relu(t)

      # hidden linear layer 
      t=self.fc2(t)
      t=F.relu(t)
      #output layer 
      t=self.out(t)
      #t=F.softmax(t,dim=1) this is not needed since we are using cross entopy loss function
       
      return t

# now lets see if the model preducts something.This is just a trial to see if the model works when we supply a sample.so we switch off the grad enabled. 
torch.set_grad_enabled(False)

#initialising the network 
 network=Network()

#obtaining a sample
sample=next(iter(train_set))
image, label=sample
print(image.shape)# we can see that the sample has a diamension of (1,28,28). But to feed a CNN network we need to have a diamension of (1,1,28,28).There should be a batch represenatation

#to add another diamension we unsqeeeze the sample 
trial_sample=image.unsqueeze(0)
print(trial_sample.shape)# now we can see that another diamension is added

#giving the sample to the model to predict 
pred=network(trial_sample)

pred.shape# means that we have one image in the batch and that has 10 predictions

# we can observe the predictions 
pred

#we can use a softmax to see the probabilities of each class 
F.softmax(pred,dim=1)

# we can use argmax to obtain the class with the higest propapalities 
F.softmax(pred,dim=1).argmax(dim=1)

print(label)

# we can observe the layers in the network
print(network.conv1)
network.conv1.weight

print(network.conv2)
network.conv2.weight

print(network.fc2)
network.fc2.weight

# we can also cheack if the network works with a batch.ititially we only input one image , now we will have a batch input and see how the output works 
#initialising the batch with 10 images 
batch_trial=torch.utils.data.DataLoader(
    train_set,
    batch_size=10

)

batch=next(iter(batch_trial))
images, labels=batch
print(images.shape)
print(labels.shape)#we can see that the shape of the images in 10,1,28,28.This mean 10 images of 1 channel which are 28 by28

#sending the images to get the prediction 
preds=network(images)

preds.shape# we can see that for every image there are 10 prediction for the classes

preds

preds.argmax(dim=1)# we can see that in all prediction the 3rd class has the highest value

#using he softmax to see the probabilities 
F.softmax(preds,dim=1).argmax(dim=1)

labels# the actual labels are totally different.This is true since we have not trained  the network

F.softmax(preds,dim=1).argmax(dim=1).eq(labels)# we can see which classes have being predicted correctly

F.softmax(preds,dim=1).argmax(dim=1).eq(labels).sum()#gets all sum of the correctly predicted

#function to obtain the correctly predicted 
def get_num_correct(predictions,labels_n):
  return predictions.argmax(dim=1).eq(labels_n).sum().item()

get_num_correct(preds,labels)

# now the traning process begins 
# the traning process inclusds the following steps in order
# get batch,pass batch to the network ,calculate the loss,calculate the gradients of the loss function with respect to the weights ,update weights with the gradients,repeat these steps untill the epoch is done, repeat all these steps to as many epochs as desired

import torch.optim as optim #this package is important for optimisation of the parameters 
torch.set_grad_enabled(True)

#reloading the train with a higher number of samples 
train_loader=torch.utils.data.DataLoader(
    train_set,
    batch_size=100

)

batch=next(iter(train_loader))#pulling one batch from the train_loader

images ,labels=batch#unpacking the images and the labels

len(images),len(labels)

# calculating thes loss
preds=network(images)
loss=F.cross_entropy(preds,labels)
loss.item()

#calculating the gradient
print(network.conv1.weight.grad)# no gradinents in our weights yet

loss.backward()# calculating the gradients

network.conv1.weight.grad.shape

network.conv1.weight.shape# the gradients and the weights have the same shape . which makes sense

#updating the wiegths 
optimizer=optim.Adam(network.parameters(),lr=0.01)#sending the paraments to the optimiser 
print(loss.item())
get_num_correct(preds,labels)# we can see that only 15 have been correclty predicted from 100

optimizer.step()#updating the weights.

# The weights of the network has been updated. We can send the same images as before and see if there is an impovement 
preds_chk=network(images)
loss=F.cross_entropy(preds_chk,labels)
loss.item()# there is a slight improvement in the loss

get_num_correct(preds_chk,labels)# still the same number of correct predictions

#Now we will train all of the above steps in one go.However this is only for one batach. ideally this should be done for all the batches and in a number of epoch.
network=Network()
train_loader=torch.utils.data.DataLoader(train_set,batch_size=100)#initialising the train loader
optimizer=optim.Adam(network.parameters(),lr=0.02)# sending all the weights in the network to the optimiser

batch=next(iter(train_loader))# loading the batch 
images,labels=batch # unloading images and labels 
preds=network(images)# obtaining the predictions 
correct_pred=get_num_correct(preds,labels)#getting the correctly predicted 
print(f'correctly predicted {correct_pred}')
loss=F.cross_entropy(preds,labels)#getting the loss 

loss.backward()# now getting the gradiensts
optimizer.step()#updating the weights 

print("loss 1:", loss.item())
preds=network(images)# reloading the sames images 
loss=F.cross_entropy(preds,labels)#checking the loss again 
print("loss after update:", loss.item())

# initiating for all epoch in and all the batches.This is the normal method to initiate the traning in one go 
network=Network()
train_loader=torch.utils.data.DataLoader(train_set,batch_size=100)#initialising the train loader
optimizer=optim.Adam(network.parameters(),lr=0.01)# sending all the weights in the network to the optimiser

total_loss=0
total_correct=0

for epoch in range(6):

    total_loss=0
    total_correct=0

    for batch in train_loader:
      images,labels=batch
      preds=network(images)# obtaining the predictions 
      loss=F.cross_entropy(preds,labels)#getting the loss 

      optimizer.zero_grad()# this is important to zero the gradients before u update other wise it will use the previous gradient values and would be a mess 
      loss.backward()# now getting the gradiensts
      optimizer.step()#updating the weights 

      total_loss+=loss.item()
      total_correct+=get_num_correct(preds,labels)

    print("epoch",epoch,"total_correct:",total_correct,"loss:",total_loss)

# doing the same thing in GPU
network=Network()
network.to(device)
train_loader=torch.utils.data.DataLoader(train_set,batch_size=100)#initialising the train loader
optimizer=optim.Adam(network.parameters(),lr=0.01)# sending all the weights in the network to the optimiser

total_loss=0
total_correct=0

for epoch in range(6):

    total_loss=0
    total_correct=0

    for batch in train_loader:
      images,labels=batch
      images=images.to(device)
      labels = labels.to(device)
      preds=network(images)# obtaining the predictions 
      loss=F.cross_entropy(preds,labels)#getting the loss 

      optimizer.zero_grad()# this is important to zero the gradients before u update other wise it will use the previous gradient values and would be a mess 
      loss.backward()# now getting the gradiensts
      optimizer.step()#updating the weights 

      total_loss+=loss.item()
      total_correct+=get_num_correct(preds,labels)

    print("epoch",epoch,"total_correct:",total_correct,"loss:",total_loss)

total_correct/len(train_set)# we have an accuracy of 88%

