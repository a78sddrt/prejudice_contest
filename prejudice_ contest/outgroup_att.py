#---------------------------------- ReadMe ---------------------------------------------

# Before running the following code, please ensure you have installed the required modules.
# You can install them using pip by running the following command in your terminal:

#pip install numpy pandas scikit-learn torch
### pip3 install numpy pandas scikit-learn torch

# Import modules
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from torch import optim
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Please modify the filename and the path of your data files.
# Two locations need to be modified:  

# 1. Replace 'YOUR PATH' with the actual path to the training data.
df_train = pd.read_csv('train.csv')

# 2. Replace 'YOUR PATH' with the actual path to your test data.
df_test = pd.read_csv('train.csv')

#---------------------------------------------------------------------------------------

data_train=df_train.loc[:,['outgroup_att','b5a','symbolic','generalized', 'identification_sol','identification_sat','identification_cen','contact_quality','contact_friendsz','disgust_p','disgust_s','disgust_r']]
data_test=df_test.loc[:,['outgroup_att','b5a','symbolic','generalized', 'identification_sol','identification_sat','identification_cen','contact_quality','contact_friendsz','disgust_p','disgust_s','disgust_r']]
train_val=data_train.values
test_val=data_test.values


#Shuffle the train data
np.random.seed(25)
np.random.shuffle(train_val)

X_train=train_val[:,1:]
y_train=train_val[:,0]
X_test=test_val[:,1:]
y_test=test_val[:,0]


train_num=len(X_train)
test_num=len(X_test)

X_train=X_train.astype(float)
X_test=X_test.astype(float)
y_train=y_train.astype(float)
y_test=y_test.astype(float)

X_train=torch.Tensor(X_train)
X_test=torch.Tensor(X_test)
y_train=torch.Tensor(y_train)
y_test=torch.Tensor(y_test)

train_data=TensorDataset(X_train, y_train)
test_data=TensorDataset(X_test, y_test)

size=10
load_train=DataLoader(train_data, batch_size=size, shuffle=True)
load_test=DataLoader(test_data, batch_size=size, shuffle=False) 

torch.manual_seed(32)
var_num=len(X_train[0])

#Definition of our model
class Quadratic(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.b=torch.nn.Parameter(torch.randn(()))
        self.W=nn.Parameter(torch.randn(1,var_num))
        self.Q=torch.nn.Parameter(torch.randn((var_num,var_num)))


    def forward(self, X):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.b+torch.matmul(self.W,X.T)+torch.diag(torch.matmul(torch.matmul(X,self.Q),X.T))

#Build a quadratic model
model=Quadratic()

#Loss function
loss_fn=nn.L1Loss()
#loss_fn=nn.MSELoss()
#loss_fn=nn.SmoothL1Loss()
#loss_fn=nn.HuberLoss()

class LogCoshLoss(torch.nn.Module):

    def __init__(self):

        super(LogCoshLoss, self).__init__()



    def forward(self, x, target):
        ey_t =target-x
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))

        
#loss_fn=LogCoshLoss()

#Choose an optimizer(here we use Adam)
optimizer=optim.Adam(model.parameters(), lr=0.0002,weight_decay=0.1)


#Training function
def train(epoch):
    model.train()

    for data, targets in load_train:
        optimizer.zero_grad()
        outputs=model(data)
        outputs=outputs.reshape(size,)
        loss=loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

    print ("epoch {}: Finished\n".format(epoch))
    
    model.eval()
    correct=0
    
    y_train_pred = model(X_train)

    y_train_pred = y_train_pred.detach().numpy()
    y_train_pred=y_train_pred.reshape(train_num,)

    y_true = y_train.numpy()
    
    r2 = r2_score(y_true, y_train_pred)
    print("Train R^2 score:", r2)

    y_test_pred = model(X_test)

    y_test_pred = y_test_pred.detach().numpy()
    y_test_pred=y_test_pred.reshape(test_num,)

    y_true = y_test.numpy()

    r2 = r2_score(y_true, y_test_pred)
    print("Test R^2 score:", r2)
    

for epoch in range(1000):
    train(epoch)

#Calculate R^2 score for testing set
y_test_pred = model(X_test)
y_test_pred = y_test_pred.detach().numpy()
y_test_pred=y_test_pred.reshape(test_num,)
y_true = y_test.numpy()

r2 = r2_score(y_true, y_test_pred)

#Print Model 
features=data_train.columns
print(features[1:])
print(model.b)
print(model.W)
print(model.Q)
b=model.b
w=model.W
q=model.Q
s=features[0]+'='+str(round(b.item(),2))
for i in range(1,len(features)):
    s+=str(round(w[0][i-1].item(),2))+"*"+features[i]+"+"

for i in range(1,len(features)):
    for j in range(1,len(features)):
        s+=str(round(q[i-1][j-1].item(),2))+"*"+features[i]+"*"+features[j]+"+"
s=s[:-1]
print(s)

#Adjusted R^2
adjust_r2=1-(1-r2)*((len(X_test)-1)/(len(X_test)-len(X_test[0])-1))
print("Adjusted R^2 score of test data:", adjust_r2)


#Calculate RMSE score for testing set
y_pred_test= model(X_test).detach().numpy()
y_test_numpy=y_test.detach().numpy()
y_pred_test=y_pred_test.reshape(test_num,)

rmse = mean_squared_error(y_test_numpy, y_pred_test, squared=False)
print(f"Test data RMSE: {rmse}")
        
