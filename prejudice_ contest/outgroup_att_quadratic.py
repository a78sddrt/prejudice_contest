#---------------------------------- ReadMe ---------------------------------------------

# Before running the following code, please ensure you have installed the required modules.
# You can install them using pip by running the following command in your terminal:

#pip install numpy pandas scikit-learn torch
#Or pip3 install numpy pandas scikit-learn torch

# Import modules
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from torch import optim
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Please modify the filename and the path of your data file.
# Replace 'YOUR PATH' with the actual path to your test data.
df_test = pd.read_csv('YOUR PATH')

#---------------------------------------------------------------------------------------

#This is model for outgroup attitude
data_test=df_test.loc[:,['outgroup_att','b5a',
                         'symbolic','generalized',
                         'identification_sol','identification_sat','identification_cen',
                         'contact_quality','contact_friendsz',
                         'disgust_p','disgust_s','disgust_r']]
test_val=data_test.values

X_test=test_val[:,1:]
y_test=test_val[:,0]

test_num=len(X_test)

X_test=X_test.astype(float)
y_test=y_test.astype(float)

X_test=torch.Tensor(X_test)
y_test=torch.Tensor(y_test)

test_data=TensorDataset(X_test, y_test)

size=10 
load_test=DataLoader(test_data, batch_size=size, shuffle=False) 

torch.manual_seed(32)


#Definition of our model
class Quadratic(torch.nn.Module):
    def __init__(self):
    
        super().__init__()
        self.b=torch.nn.Parameter(torch.tensor(0.0352, requires_grad=True))
        self.W=nn.Parameter(torch.tensor([[ 0.1088, -0.2675, -0.1069,  0.0799,  0.0322,  0.0281,  0.4734,  0.1871,
         -0.0153,  0.0673,  0.0897]],
       requires_grad=True))
        self.Q=torch.nn.Parameter(torch.tensor(
        [[ 1.9905e-02, -1.9163e-02, -4.1446e-03, -1.7958e-02, -4.0608e-03,
          9.6276e-03, -2.3313e-02,  1.3277e-02,  2.6977e-03,  6.7479e-03,
         -1.7870e-03],
        [-1.9163e-02, -1.8359e-02, -1.4176e-03, -2.7873e-03, -2.0722e-03,
          2.4892e-03, -1.8234e-02,  7.4995e-03, -5.6516e-03,  1.8411e-02,
          1.2780e-02],
        [-4.1464e-03, -2.7420e-03, -1.8914e-02,  7.5234e-03, -7.0477e-03,
         -2.2964e-02, -2.2197e-02, -6.5652e-04,  1.5492e-02,  8.4251e-03,
         -8.9469e-03],
        [-1.7958e-02, -1.5106e-03, -4.9565e-03,  3.9077e-02,  1.9528e-02,
         -4.1614e-02,  1.9691e-02,  4.6086e-03, -3.5720e-02, -2.8080e-02,
         -4.3565e-03],
        [-4.0609e-03, -5.8599e-05,  2.8437e-03,  1.9610e-02,  1.2490e-02,
          6.2798e-03,  8.5521e-03,  2.9581e-03,  2.2397e-03,  3.4000e-03,
          9.4444e-03],
        [ 9.6276e-03,  2.5803e-03, -2.4021e-02, -4.2048e-02,  6.3566e-03,
          3.0543e-02,  2.3951e-03,  1.3582e-02,  3.8172e-02,  1.0154e-02,
         -4.1583e-02],
        [-2.3313e-02, -2.0075e-02, -2.1060e-02,  1.9819e-02,  8.5894e-03,
          2.3877e-03, -4.4822e-03, -5.0609e-03,  3.0793e-02,  1.4580e-02,
          1.4105e-02],
        [ 1.3277e-02,  7.2998e-03, -2.2733e-03,  4.6163e-03,  2.9685e-03,
          1.3584e-02, -5.0574e-03, -2.4532e-02, -1.6906e-02, -1.9561e-02,
          5.4819e-03],
        [ 2.6977e-03, -5.6515e-03,  1.5493e-02, -3.5720e-02,  2.2397e-03,
          3.8172e-02,  3.0793e-02, -1.6906e-02,  6.4228e-02, -6.8594e-03,
          2.2768e-02],
        [ 6.7479e-03,  1.8403e-02,  8.4628e-03, -2.8085e-02,  3.3962e-03,
          1.0154e-02,  1.4581e-02, -1.9561e-02, -6.8594e-03, -8.6864e-03,
         -2.5171e-02],
        [-1.7870e-03,  1.2780e-02, -8.9544e-03, -4.3565e-03,  9.4444e-03,
         -4.1583e-02,  1.4105e-02,  5.4819e-03,  2.2768e-02, -2.5171e-02,
          3.3194e-02]], requires_grad=True))


    def forward(self, X):
        return self.b+torch.matmul(self.W,X.T)+torch.diag(torch.matmul(torch.matmul(X,self.Q),X.T))

#Build model
model=Quadratic()


# Calculate R^2 score for test data
y_test_pred = model(X_test)
y_test_pred = y_test_pred.detach().numpy()
y_test_pred=y_test_pred.reshape(test_num,)
y_true = y_test.numpy()

r2 = r2_score(y_true, y_test_pred)

#Adjusted R^2
adjust_r2=1-(1-r2)*((len(X_test)-1)/(len(X_test)-len(X_test[0])-1))
print("Adjusted R^2 score of test data (outgroup_att):", adjust_r2)


# Calculate RMSE score for test data
y_pred_test= model(X_test).detach().numpy()
y_test_numpy=y_test.detach().numpy()
y_pred_test=y_pred_test.reshape(test_num,)

rmse = mean_squared_error(y_test_numpy, y_pred_test, squared=False)
print(f"Test data RMSE (outgroup_att): {rmse}")


#Print Model
print("Bias Model:")
print(model.b)
print(model.W)
print(model.Q)


        
