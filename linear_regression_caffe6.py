#%%  0.Imports
from caffe2.python import brew, model_helper, optimizer, workspace

#%%  1. Create input data (in numpy)
x,y = load_data()
#%%  2.Create input tensors and feed them data
model = model_helper.ModelHelper("model")
X = model.StopGradient("X")
Y = model.StopGradient("Y")
x,y = load_data()
workspace.FeedBlob("X", x)
workspace.FeedBlob("Y", y)
#%% 2.Create neural network(1 neuron)
pred = brew.fc(model, X, "pred",1,1)
loss = model.net.SquaredL2Distance([pred, Y], "loss")

#3.Select An Optimizer to minimize loss
model.AddGradientOperators([loss])
opt = optimizer.build_sgd(model, base_learning_rate=1e-5)
for param in model.GetOptimizationParamInfo():
    opt(model.net, model.param_init_net, param)

#4. Init session/variables/net
workspace.RunNetOnce(model.param_init_net)  
workspace.CreateNet(model.net)              

#5.Training Loop
num_epochs = 20000
losses =[]
start_time = time.time()

for e in range(num_epochs):
  workspace.RunNet(model.net, 10)         
  if e%100 == 0 :
      print("Epoch %s; loss: %s" %(e,workspace.FetchBlob("loss").sum()))
end_time = time.time()






#%%
print("Time"+str(end_time-start_time))
plt.plot(losses)
plt.show()
plt.plot(x,y,"*b")
plt.plot(x,workspace.FetchBlob("y_pred"),".r")
plt.show()
