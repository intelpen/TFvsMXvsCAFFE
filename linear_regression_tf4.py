import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
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
tf.reset_default_graph()

#1.Create Placeholders for input tensors
X = tf.placeholder(shape = [None,1], dtype = tf.float32)
Y = tf.placeholder(shape = [None,1],dtype = tf.float32)
x,y = load_data()
feed_dict = {X:x, Y:y}



#2.Define network architecture and loss 
y_pred =  tf.layers.dense(X, 1)
loss = tf.losses.mean_squared_error(Y,y_pred)

#3.Select An Optimizer to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.000001)
train_step = optimizer.minimize(loss)



#4. Init session/variables/net
sess= tf.Session() 
sess.run(tf.global_variables_initializer())

#5.Training Loop
num_epochs = 20000
losses =[]
start_time = time.time()

for e in range(num_epochs):
  _,output, loss_epoch = sess.run(fetches = [train_step,y_pred,loss], feed_dict = feed_dict)
  losses.append(loss_epoch)
  if e%100 ==0:
    print("Epoch %s; loss: %s" %(e,loss_epoch))
end_time = time.time()





#%%
print("Time"+str(end_time-start_time))
plt.plot(losses)
plt.show()
plt.plot(x,y,"*b")
plt.plot(x,output,".r")
plt.show()

ker= tf.get_default_graph().get_tensor_by_name("dense/kernel:0")
ker_out = sess.run(ker)
print(ker_out)
bias= tf.get_default_graph().get_tensor_by_name("dense/bias:0")
bias_out = sess.run(bias)
print(bias_out)