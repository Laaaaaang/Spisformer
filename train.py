import tran
import numpy as np
import torch 
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from data_process import val_loader
from data_process import train_loader
from data_process import test_loader
from data_process import testY
from data_process import scaler2
import data_process
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

#hyperparameters
look_back=60
T=1
epochs=5000       #training epochs
num_features=3     #number of features
num_heads=4     #number of heads in multiattention block
embed_dim=60   #embedding dimension
dense_dim=5    #dense dimension
dropout_rate=0.01   #dropout rate
num_blocks=3    #number of encoders and decoders
learn_rate=0.001    #learning rate
batch_size=200   #batch size
kernel_size = 1
pool_size = 3 #for dot product
conv_out_channels = 60
print(torch.cuda.is_available())

model = tran.Transformer(num_features=num_features, embed_dim=embed_dim, dense_dim=dense_dim, num_heads=num_heads, dropout_rate=dropout_rate, num_blocks=num_blocks, output_sequence_length=T, conv_out_channels=conv_out_channels,kernel_size=kernel_size, pool_size=pool_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#defeine loss function
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

train_losses = []
val_losses = []
start_time = time.time()
#trianing loop
for epoch in range(epochs):
    model.train()
    for inputs,labels in tqdm(train_loader,position=0):
        inputs = inputs.to(device)
        # print(inputs)
        # print(inputs.shape)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader ,position=0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model.forward(inputs)
            val_loss = criterion(outputs, labels)
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())   
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

end_time = time.time()
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


#model testing
model.eval()
predictions = []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, position=0):
        inputs = inputs.to(device)
        outputs = model.forward(inputs)
        predictions.extend(outputs.cpu().numpy())

predictions = np.array(predictions).reshape(-1,1)
labels = (testY.cpu().numpy()).reshape(-1,1)

predictions = scaler2.inverse_transform(predictions)
labels = scaler2.inverse_transform(labels)

#model evaluation
r2 = r2_score(labels, predictions)
mae = mean_absolute_error(labels,predictions)
rmse = np.sqrt(mean_squared_error(labels,predictions))
mape = np.mean(np.abs((labels - predictions) / labels))
total_time = end_time - start_time
#print all the evaluation parameters
print('R2:',r2)
print('MAE:',mae)
print('RMSE:',rmse)
print('MAPE:',mape)
print('Total_Time',total_time)
#plot final result
plt.xlabel('time/s', fontsize=13)
plt.ylabel('current', fontsize=13)
plt.plot(labels, label='Real')
plt.plot(predictions, label='Prediction')
plt.legend(fontsize=20)
plt.show()
