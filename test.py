import torch
import numpy as np
import warnings
from ConvolutionalNetwork import ConvolutionalNetwork
warnings.filterwarnings("ignore")

def test_func(X):
    X=X.reshape(X.shape[0],1,150,150)
    X=X/255
    X_testTensor = torch.Tensor(X)
    y_pred=model(X_testTensor)
    predicted = torch.max(y_pred.data, 1)[1]
    return predicted
    
model_path="model.pth"

model=ConvolutionalNetwork()
model.load_state_dict(torch.load(model_path))
model.eval()
model.state_dict()

X = np.load('Test_Images.npy')
y_test=np.load('Test_Labels.npy')
predicted = test_func(X)
correct=(predicted.numpy() == y_test).sum()
accuracy=correct/y_test.shape*100
print('Accuracy = ',accuracy)