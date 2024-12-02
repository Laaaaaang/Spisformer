import numpy as np
import torch 
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

#hyperparameters
look_back=60
T=1
epochs=100       #training epochs
num_features=3  #number of features
num_heads=4     #number of heads in multiattention block
embed_dim=60    #embedding dimension
dense_dim=64    #dense dimension
dropout_rate=0.01   #dropout rate
num_blocks=3    #number of encoders and decoders
cnn_blocks=3
learn_rate=0.001    #learning rate
batch_size=300   #batch size
conv_out_channels = 60
kernel_size = 1
pool_size = 3  #for the same size of tensor in order to dot product

#main function transformer
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate):
        super(TransformerEncoder,self).__init__()

        self.mha = nn.MultiheadAttention(embed_dim,num_heads)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.dense1 = nn.Linear(embed_dim,dense_dim)
        self.dense2 = nn.Linear(dense_dim,embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
    
    def forward(self,inputs):
        attn_output,_ = self.mha(inputs,inputs,inputs)#this step refers to the QKV matrix input
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        dense_output = self.dense1(out1)
        dense_output = self.dense2(dense_output)
        dense_output = self.dropout2(dense_output)
        out2 = self.layernorm2(out1+dense_output)

        return out2
    
#define class decoder
class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate):
        super(TransformerDecoder,self).__init__()

        self.mha1 = nn.MultiheadAttention(embed_dim,num_heads)
        self.mha2 = nn.MultiheadAttention(embed_dim,num_heads)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.layernorm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.dense1 = nn.Linear(embed_dim, dense_dim)
        self.dense2 = nn.Linear(dense_dim, embed_dim)
        self.layernorm4 = nn.LayerNorm(embed_dim)
        self.dropout4 = nn.Dropout(dropout_rate)

    def forward(self, inputs, encoder_outputs):
        attn1, _ = self.mha1(inputs,inputs,inputs)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(inputs + attn1)

        attn2, _ = self.mha2(out1,encoder_outputs,encoder_outputs)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)

        dense_output = self.dense1(out2)#inside dense we have relu etc. nonlinear function
        dense_output = self.dense2(dense_output)
        dense_output = self.dropout3(dense_output)
        out3 = self.layernorm3(out2 + dense_output)

        decoder_output = self.dense1(out3)#we have another dense layer for scalibility
        decoder_output = self.dense2(decoder_output)
        decoder_output = self.dropout4(decoder_output)
        out4 = self.layernorm4(out3 + decoder_output)
        return out4
    
class CNN(nn.Module):
    def __init__(self, embed_dim, dense_dim, conv_out_channels, kernel_size, pool_size):
        super(CNN, self).__init__()
        # # 定义 embedding 层，将输入映射到嵌入维度
        self.embedding = nn.Linear(embed_dim, embed_dim)
        
        # 定义卷积层，使用 embed_dim 作为输入通道，指定输出通道和卷积核大小
        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=conv_out_channels, kernel_size=kernel_size)
        
        # 定义平均池化层，指定池化窗口大小
        self.pool = nn.AvgPool1d(pool_size)
        
        # 定义全连接层
        self.dense1 = nn.Linear(conv_out_channels, dense_dim)
        self.dense2 = nn.Linear(dense_dim, embed_dim)
    
    def forward(self, inputs):
        # # 展平张量以适应全连接层
        # inputs = inputs.view(inputs.size(0), -1)  # 或者 CNN_out3.flatten(start_dim=1)
        # # 输入通过 embedding 层
        # print(inputs.shape)
        CNN_out1 = inputs
        # print(CNN_out1.shape)
        # 改变输入形状为 (batch_size, channels, sequence_length) 以便 Conv1d 层处理
        
        # CNN_out1 = CNN_out1.permute(0,2,1)
        # print(CNN_out1.shape)
        # 通过卷积层
        CNN_out2 = self.conv(CNN_out1)
        # print(CNN_out2.shape)
        # # 通过池化层
        # CNN_out3 = self.pool(CNN_out2)
        # print(CNN_out3.shape)
        # # 还原形状以便全连接层处理
        # CNN_out3 = CNN_out3.view(CNN_out3.size(0), -1)
        # print(CNN_out3.shape)
        # 通过第一个全连接层
        CNN_out4 = self.dense1(CNN_out2)
        # print(CNN_out4.shape)
        # 通过第二个全连接层
        CNN_output = self.dense2(CNN_out4)
        # print(CNN_output.shape)
        return CNN_output

class Transformer(nn.Module):
    def __init__(self, num_features,embed_dim, dense_dim, num_heads, dropout_rate, num_blocks, output_sequence_length, conv_out_channels, kernel_size, pool_size):
        super(Transformer,self).__init__()

        self.embedding = nn.Linear(num_features, embed_dim)
        self.cnn_embedding = nn.Linear(embed_dim,embed_dim)
        self.transformer_encoder = nn.ModuleList([TransformerEncoder(embed_dim, dense_dim, num_heads, dropout_rate) for _ in range(num_blocks)])
        self.transformer_decoder = nn.ModuleList([TransformerDecoder(embed_dim, dense_dim, num_heads, dropout_rate) for _ in range(num_blocks)])
        self.cnn_block = nn.ModuleList([CNN(embed_dim, dense_dim, conv_out_channels, kernel_size, pool_size)for _ in range(num_blocks)])
        self.final_layer = nn.Linear(embed_dim * look_back, output_sequence_length)

    def forward(self, inputs):
        encoder_inputs = inputs
        decoder_inputs = inputs
        # print(inputs.shape)
        encoder_outputs = self.embedding(encoder_inputs)
        for i in range(len(self.transformer_encoder)):
            encoder_outputs = self.transformer_encoder[i](encoder_outputs)

        # print(inputs.shape)
        cnn_inputs = self.embedding(inputs)
        # print(cnn_inputs.shape)
        for i in range(len(self.cnn_block)):
            cnn_outputs = self.cnn_block[i](cnn_inputs)
            # decoder_outputs = self.transformer_decoder[i](decoder_outputs, encoder_outputs)
        
        # print("cnn","encoder",cnn_outputs.shape,encoder_outputs.shape)
        combined_outputs = cnn_outputs * encoder_outputs # dot product to augment the memory gain from short-term(CNN) and long-term(transformer)
        # print(encoder_outputs.shape)

        decoder_outputs = self.embedding(decoder_inputs)
        for i in range(len(self.transformer_decoder)):
            decoder_outputs = self.transformer_decoder[i](decoder_outputs,combined_outputs)
            

        decoder_outputs = decoder_outputs.view(-1,decoder_outputs.shape[1] * decoder_outputs.shape[2])
        decoder_outputs = self.final_layer(decoder_outputs)
        decoder_outputs = decoder_outputs.view(-1,T)
        return decoder_outputs
    




