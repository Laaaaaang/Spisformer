import numpy as np
import torch 
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# Hyperparameters
look_back = 5
T = 1
epochs = 100       # training epochs
num_features = 5   # number of features
num_heads = 4      # number of heads in multiattention block
embed_dim = 32     # embedding dimension
dense_dim = 64     # dense dimension
dropout_rate = 0.01   # dropout rate
num_blocks = 3     # number of encoders and decoders
learn_rate = 0.001    # learning rate
batch_size = 300   # batch size

# Read RC information (from CSV file)
rc_matrix = pd.read_csv('RC_1.csv', header=None).values.flatten()  # Flatten to 1D
# print(rc_matrix)
# Read current-time matrix (from XLSX file)
dataset = pd.read_excel('processed_result_1.xlsx', usecols=[0, 1, 2, 3])  # Assuming 4 columns of features
# print(dataset)
dataY = dataset['Current'].values
# print(dataY)

# Data processing
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaler2 = MinMaxScaler(feature_range=(0, 1))
data_X = scaler1.fit_transform(dataset.drop(columns=['Current']).values)
# print(len(data_X))
data_Y = scaler2.fit_transform(dataY.reshape(-1, 1))
# print(len(data_Y))
# Train/validation/test split
train_size = int(len(data_X) * 0.7)
val_size = int(len(data_X) * 0.2)
test_size = len(data_X) - train_size - val_size

train_X, train_Y = data_X[:train_size], data_Y[:train_size]
val_X, val_Y = data_X[train_size:train_size + val_size], data_Y[train_size:train_size + val_size]
test_X, test_Y = data_X[train_size + val_size:], data_Y[train_size + val_size:]

# Create dataset function (with static RC matrix)
def create_dataset(datasetX, datasetY, look_back=1, T=1):
    dataX, dataY = [], []
    for i in range(0, len(datasetX) - look_back - T, T):
        a = datasetX[i:(i + look_back), :]
        dataX.append(a)
        if T == 1:
            dataY.append(datasetY[i + look_back])
        else:
            dataY.append(datasetY[i + look_back:i + look_back + T, 0])
    return np.array(dataX), np.array(dataY)

# Prepare training dataset
trainX, trainY = create_dataset(train_X, train_Y, look_back, T)
# print(rc_matrix.shape)
valX, valY = create_dataset(val_X, val_Y, look_back, T)
testX, testY = create_dataset(test_X, test_Y, look_back, T)

# Convert to tensor
trainX = torch.Tensor(trainX)
trainY = torch.Tensor(trainY)
valX = torch.Tensor(valX)
valY = torch.Tensor(valY)
testX = torch.Tensor(testX)
testY = torch.Tensor(testY)
rc = torch.Tensor(rc_matrix)
# print(rc.shape)
rc = rc.unsqueeze(1)
rc = rc.expand(rc.shape[0],trainX.shape[2])
# print(rc.shape)
# Define Dataset
class MyDataset(Dataset):
    def __init__(self, data_X, data_Y, rc_matrix):
        self.data_X = data_X
        self.data_Y = data_Y
        self.rc = rc_matrix  # Convert RC matrix to tensor
        # self.data_X = torch.cat((self.rc,self.data_X),dim=0)
        #         # 扩展 y 张量的维度到 [n, 1, 1] 以便进行连接
        # y_expanded = y.unsqueeze(1).unsqueeze(2)  # 结果大小 [n, 1, 1]

        # # 扩展 y_expanded 使其匹配 x 的第二和第三个维度
        # y_expanded = y_expanded.expand(n, b, c)   # 结果大小 [n, b, c]

        # # 连接 x 和 y_expanded
        # result = torch.cat((y_expanded, x), dim=0)  # 沿着 a 维连接，结果大小 [a+n, b, c]
    def __getitem__(self, index):
        x = self.data_X[index]
        y = self.data_Y[index]
        x = torch.cat((rc,x),dim=0)
        # Concatenate RC matrix with the dynamic input (current row)
        return x, y

    def __len__(self):
        return len(self.data_X)

# Prepare data loaders
train_dataset = MyDataset(trainX, trainY, rc)
val_dataset = MyDataset(valX, valY, rc)
test_dataset = MyDataset(testX, testY, rc)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
