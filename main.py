import tran
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from data_process import val_loader
from data_process import train_loader
import data_process
import tran
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from data_process import val_loader
from data_process import train_loader
import data_process
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from config import epochs, learn_rate, batch_size, num_features, embed_dim, dense_dim, num_heads, dropout_rate, num_blocks,T
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
from config import epochs, learn_rate, batch_size, num_features, embed_dim, dense_dim, num_heads, dropout_rate, num_blocks,T
from train import train_model, evaluate_model
criterion = nn.MSELoss()


# 加载模型并使用GPU
model = tran.Transformer(num_features=num_features, embed_dim=embed_dim, dense_dim=dense_dim, num_heads=num_heads, dropout_rate=dropout_rate, num_blocks=num_blocks, output_sequence_length=T)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)


#train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, epochs,device)
#这是训练函数，可以进行训练。
#如果要直接加载模型，可以不用这个函数，用下面的model.load_state_dict(torch.load('model.pth'))，修改model.pth名字即可


# 加载已训练好的模型参数
try:
    model.load_state_dict(torch.load('model_epoch_50.pth'))  # 加载模型参数
    print("模型参数加载成功。")
    model.eval()  # 将模型设置为评估模式
except FileNotFoundError:
    print("未找到模型文件。")

# 这是评估函数，可以使用测试集对一个模型进行测试，并画出图像
evaluate_model(model, data_process.test_loader, device)




