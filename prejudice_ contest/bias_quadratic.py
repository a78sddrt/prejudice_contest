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
df_test = pd.read_csv('train.csv')

#---------------------------------------------------------------------------------------

#This is model for bias
data_test=df_test.loc[:,['bias','b5a',
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
        self.b=torch.nn.Parameter(torch.tensor(-0.0906, requires_grad=True))
        self.W=nn.Parameter(torch.tensor([[ 0.0335,  0.1836,  0.2010,  0.3206,  0.1490, -0.0064, -0.3185, -0.1422,
         -0.0207, -0.0216, -0.0763]],
       requires_grad=True))
        self.Q=torch.nn.Parameter(torch.tensor(
        [[-4.0944e-02,  2.0122e-02,  8.7684e-04,  1.5676e-04,  1.7145e-02,
         -3.0407e-02,  1.4269e-02,  1.2964e-02, -7.9995e-03, -2.9237e-03,
         -7.2917e-03],
        [ 2.0426e-02,  4.3515e-03,  4.1600e-03, -1.4977e-02, -2.7929e-02,
         -4.4473e-03,  3.0062e-02,  6.0947e-03,  1.1220e-02, -2.7037e-02,
          1.5781e-02],
        [-2.1417e-04, -6.4160e-03,  1.5398e-02,  1.3346e-01, -1.8155e-01,
          3.3097e-02, -2.4173e-03,  4.2235e-02, -1.9313e-02,  8.2293e-03,
         -2.0883e-03],
        [ 2.7649e-04,  2.6325e-02, -1.1239e-01,  3.8708e-02, -3.1324e-02,
          2.4045e-02, -2.7013e-02,  2.1834e-03,  1.1414e-02,  8.8395e-03,
          3.3530e-02],
        [ 1.6431e-02,  6.3874e-02,  1.2963e-01, -2.8082e-02,  3.3065e-02,
         -2.0117e-02, -1.7458e-02,  1.7810e-02,  1.1791e-02, -4.0892e-03,
         -3.5256e-02],
        [-3.0381e-02,  2.9224e-03, -2.0965e-02,  3.9679e-04, -8.5728e-03,
          2.8275e-02,  7.7793e-03, -2.6672e-02, -2.9451e-02,  1.9268e-03,
         -3.8786e-03],
        [ 1.4236e-02, -1.2413e-02,  1.9493e-02, -1.1779e-02, -1.1692e-02,
          5.7876e-03,  4.2145e-02, -8.4391e-03, -2.8751e-02,  1.1688e-02,
         -8.5683e-04],
        [ 1.2923e-02, -8.6956e-03, -5.5679e-02,  4.0601e-03,  2.2269e-02,
         -2.5586e-02, -6.7520e-03, -6.6175e-03,  3.0278e-02, -8.8328e-03,
          1.6301e-02],
        [-7.9995e-03,  1.1455e-02, -1.5767e-02,  1.1443e-02,  1.2025e-02,
         -2.9452e-02, -2.8750e-02,  3.0279e-02, -5.8731e-02,  2.2674e-02,
         -4.1721e-02],
        [-2.9409e-03, -2.9604e-02,  1.3385e-02,  6.1960e-03, -8.0210e-03,
          1.4607e-03,  1.3863e-02, -7.8828e-03,  2.2671e-02, -4.4016e-02,
          4.4512e-02],
        [-7.2917e-03,  1.5290e-02, -6.4066e-03,  3.3704e-02, -3.5001e-02,
         -3.8808e-03, -9.1356e-04,  1.6326e-02, -4.1721e-02,  4.4501e-02,
         -4.2902e-02]], requires_grad=True))


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
print("Adjusted R^2 score of test data (Bias):", adjust_r2)


# Calculate RMSE score for test data
y_pred_test= model(X_test).detach().numpy()
y_test_numpy=y_test.detach().numpy()
y_pred_test=y_pred_test.reshape(test_num,)

rmse = mean_squared_error(y_test_numpy, y_pred_test, squared=False)
print(f"Test data RMSE (Bias): {rmse}")


#Print Model
print(model.b)
print(model.W)
print(model.Q)


        
