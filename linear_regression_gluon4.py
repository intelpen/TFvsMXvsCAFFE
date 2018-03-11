from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd,autograd, gluon
import matplotlib.pyplot as plt
import time
#%%
def load_data(num_examples =200):
    x = np.random.random((num_examples,1)).astype(np.float32)*20-10
    noise = np.random.random((num_examples,1)).astype(np.float32)*3
    y= x*2.2 + 10 +noise
    plt.plot(x,y,"*")
    plt.show()
    return x,y
#%%


#1.Create Placeholders for input tensors
ctx =mx.gpu()
x,y =load_data()
X = nd.array(x, ctx = ctx)
Y = nd.array(y, ctx = ctx)



#2.Define network architecture and loss 
pred = gluon.nn.Dense(units=1,in_units =1) 
square_loss = gluon.loss.L2Loss()# Define loss

#3.Select An Optimizer to minimize loss
pred.collect_params().initialize(mx.init.Normal(sigma = 1.0), ctx = ctx) #this is needed before trainer
# Optimizer
trainer = gluon.Trainer(pred.collect_params(),'sgd', {'learning_rate':0.000001})


#4. Init session/variables/net - was done in step 3
# Initialize parameters 


#5.Training Loop
num_epochs = 20000
losses = []
start_time = time.time()

for e in range(num_epochs):
  with autograd.record(): 
    output = pred(X)
    loss = square_loss(output, Y)
  loss.backward()
  trainer.step(batch_size=2000)
  epoch_loss = nd.mean(loss).asscalar()
  if (e%100 ==0):
    print("Epoch %s; loss: %s" %(e,epoch_loss))
  losses.append(epoch_loss)
end_time = time.time()
#%%
print("Time"+str(end_time-start_time))
# Visualizing the learning curve
plt.plot(losses)
plt.show()
plt.plot(x,y,"*b")
plt.plot(x,output.asnumpy(),".r")
plt.show()
# Getting the learned model parameters

params = net.collect_params()

for param in params.values():
  print(param.name, param.data())


