import math
# PyTorch
import torch
from torchvision import models
from torch import cuda 
import torch.nn as nn
# warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# Data science tools
import numpy as np
from selfonn import SelfONNLayer

def reset_function_generic(m):
    if hasattr(m,'reset_parameters') or hasattr(m,'reset_parameters_like_torch'): 
        # print(m) 
        if isinstance(m, SelfONNLayer):
            m.reset_parameters_like_torch() 
        else:
            m.reset_parameters()

class SqueezeLayer(nn.Module):
    
    def forward(self,x):
        x = x.squeeze(2)
        x = x.squeeze(2)
        return x 

class UnSqueezeLayer(nn.Module):
    
    def forward(self,x):
        x = x.unsqueeze(2).unsqueeze(3)
        return x 

class ModifiedMobileNet_V2(nn.Module):
  #mobileSelfONNet
  def __init__(self, model):
    super().__init__()
    self.model = model
    self.mobileHead =nn.Sequential(
          *list(self.model.features.children())[:-1])
    # self.adapt = nn.AdaptiveMaxPool2d((2,2))
    self.flat = torch.nn.Flatten()
    self.n_input=0
    
  def forward(self,x):
    x=self.mobileHead(x)
    # x=self.adapt(x)
    x=self.flat(x)
    self.n_inputs=x.size(1)

    return  x



class CNN_LSTM(nn.Module):
    def __init__(self,baseModel, num_features, LSTM_hidden_size, LSTM_num_layers, LSTM_drop_rate,bidirectional, class_num,q_order):
        super().__init__()
        self.num_features = num_features
        self.LSTM_hidden_size = LSTM_hidden_size
        self.LSTM_num_layers = LSTM_num_layers
        self.LSTM_drop_rate = LSTM_drop_rate
        self.class_num = class_num
        self.q_order=q_order
        self.baseModel = baseModel
        self.rnn = nn.LSTM(num_features, LSTM_hidden_size, LSTM_num_layers, batch_first=True, dropout=LSTM_drop_rate,bidirectional=bidirectional) 
        if bidirectional:
            self.fc = MLP_Classifier(in_channels=2*LSTM_hidden_size, class_num=class_num, classifer_typ='TwoSelfMLP',q_order=self.q_order)
        else:
            self.fc = MLP_Classifier(in_channels=LSTM_hidden_size, class_num=class_num, classifer_typ='TwoSelfMLP',q_order=self.q_order)

    def forward(self, x):
        b, ts, c, h, w = x.shape
        t = 0
        y = self.baseModel(x[:, t, :, :, :])
        output, (hn, cn) = self.rnn(y.unsqueeze(1))
        for t in range(1, ts):
            y = self.baseModel(x[:, t, :, :, :])
            # input of shape Batch, Sequence, Input_size   >>>  b, 1, num_features
            # input of shape Batch, Sequence, Output_size  >>>  b, 1, LSTM_hidden_size
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        # flatten,  b, 1, LSTM_hidden_size  >> b, LSTM_hidden_size
        out = self.fc(out[:,-1]) 
        return out 

class MLP_Classifier(nn.Module):
    def __init__(self,in_channels, class_num, classifer_typ, q_order):
        super().__init__()
        
        if classifer_typ =='SingleMLP':
            # Single MLP layer
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),  
                nn.Linear(in_channels, class_num), 
                nn.LogSoftmax(dim=1) 
                ) 
            # xavier initialization 
            gain = nn.init.calculate_gain('linear')
            torch.nn.init.xavier_uniform_(self.classifier[1].weight,gain=gain) 
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.classifier[1].weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.classifier[1].bias, -bound, bound)
        elif classifer_typ =='TwoMLP':
            # Two MLP layers
            self.classifier = nn.Sequential(
                nn.Dropout(0.2), 
                nn.Linear(in_channels, 128), 
                nn.Dropout(0.2), 
                nn.ReLU(), 
                nn.Linear(128, class_num), 
                nn.LogSoftmax(dim=1) 
                )
            # xavier initialization 
            torch.nn.init.xavier_uniform_(self.classifier[1].weight)
            self.classifier[1].bias.data.fill_(0.01) 
            torch.nn.init.xavier_uniform_(self.classifier[4].weight)
            self.classifier[4].bias.data.fill_(0.01) 
        elif classifer_typ =='SingleSelfMLP':
            # Single SelfMLP layer 
            self.classifier = nn.Sequential(
                UnSqueezeLayer(),
                nn.Dropout(0.2), 
                nn.Tanh(),
                SelfONNLayer(in_channels=in_channels,out_channels=class_num,kernel_size=1,q=q_order,mode='fast',dropout=None),
                nn.Tanh(),
                SqueezeLayer(),
                nn.LogSoftmax(dim=1) 
                ) 
        elif classifer_typ =='TwoSelfMLP':
            # Two SelfMLP layer 
            self.classifier = nn.Sequential(
                UnSqueezeLayer(),
                nn.Dropout(0.2), 
                nn.Tanh(),
                SelfONNLayer(in_channels=in_channels,out_channels=32,kernel_size=1,q=q_order,mode='fast',dropout=0.2),
                nn.Dropout(0.2), 
                nn.Tanh(), 
                SelfONNLayer(in_channels=32,out_channels=class_num,kernel_size=1,q=q_order,mode='fast',dropout=None),
                SqueezeLayer(),
                nn.LogSoftmax(dim=1) 
                )

    def forward(self,x):
        x = self.classifier(x)
        return x



def cnn_V1(input_ch, class_num): 
    model = torch.nn.Sequential(
        # 1st layer (conv) 
        torch.nn.Conv2d(input_ch, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 2nd layer (conv)
        torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 3rd layer (conv)
        torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
        torch.nn.ReLU(inplace=True),
        # 4th layer (conv)
        torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 5th layer (conv)
        torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        # 6th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 7th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        # 8th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # Average pooling 
        torch.nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        torch.nn.Flatten(), 
    )  
    #
    return model 

def cnn_V2(input_ch, class_num): 
    model = torch.nn.Sequential(
        # 1st layer (conv) 
        torch.nn.Conv2d(input_ch, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 2nd layer (conv)
        torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 3rd layer (conv)
        torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
        torch.nn.ReLU(inplace=True),
        # 4th layer (conv)
        torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 5th layer (conv)
        torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        # 6th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 7th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        # 8th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # Average pooling 
        torch.nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        torch.nn.Flatten(), 
    )
    #
    return model 

def cnn_V3(input_ch, class_num): 
    model = torch.nn.Sequential(
        # 1st layer (conv)
        torch.nn.Conv2d(input_ch, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 2nd layer (conv)
        torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 3rd layer (conv)
        torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
        torch.nn.ReLU(inplace=True),
        # 4th layer (conv)
        torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 5th layer (conv)
        torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        # 6th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 7th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        # 8th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # Average pooling 
        torch.nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        torch.nn.Flatten(), 
    )
    #
    return model 


def cnn_V4(input_ch, class_num): 
    model = torch.nn.Sequential(
        # 1st layer (conv)
        torch.nn.Conv2d(input_ch, 20, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        # Average pooling 
        torch.nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        torch.nn.Flatten(), 
    )
    #
    return model 


class cnn_V5(nn.Module):
    
    def __init__(self, input_ch, class_num): 
        super(cnn_V5, self).__init__() 

        # 1st layer (conv)
        self.conv1 = cnn_V5.conv_block(in_channels=input_ch, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        # Average pooling 
        self.AvgPool = torch.nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.flatten = torch.nn.Flatten()
        # 2nd layer (MLP) 
        # conv_output = 7*7*20= 980
        self.MLP2 = torch.nn.Linear(in_features=980, out_features=class_num, bias=True)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        layer1 = self.conv1(x)
        layer1 = self.pool1(layer1)
        Pool_layer = self.AvgPool(layer1)
        Pool_layer = self.flatten(Pool_layer)
        Output_layer = self.MLP2(Pool_layer) 
        return self.softmax(Output_layer) 

    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):     
        return nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding), 
            torch.nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True) 
        )

def SelfONN_1(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(4),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(4),  
        torch.nn.Tanh(), 
        # flatten 
        torch.nn.Flatten(),  
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model 



def SelfONN_2(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),  
        torch.nn.Tanh(), 
        # 3rd layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 4th layer (conv)
        SelfONNLayer(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(4),  
        torch.nn.Tanh(), 
        # flatten 
        torch.nn.Flatten(),  
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model 


def SelfONN_3(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast',dropout=0.2),
        # torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast',dropout=0.2),
        torch.nn.MaxPool2d(3),  
        torch.nn.Tanh(), 
        # 3rd layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast',dropout=0.2),
        torch.nn.MaxPool2d(3),
        torch.nn.Tanh(),
        # 4th layer (conv)
        SelfONNLayer(in_channels=8,out_channels=12,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast',dropout=0.2),
        torch.nn.MaxPool2d(3),
        torch.nn.Tanh(), 
        # 6th layer (conv)
        SelfONNLayer(in_channels=12,out_channels=250,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast',dropout=0.2),
        torch.nn.MaxPool2d(3),  
        torch.nn.Tanh(), 
        # flatten 
        torch.nn.Flatten(),  
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model


def SelfONN_mArSL(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=16,kernel_size=7,stride=1,padding=4,dilation=1,groups=1,bias=True,q=q_order,mode='fast',dropout=0.2),
        torch.nn.Tanh(),

        # 2nd layer (conv) 
        SelfONNLayer(in_channels=16,out_channels=16,kernel_size=7,stride=1,padding=4,dilation=1,groups=1,bias=True,q=q_order,mode='fast',dropout=0.2),
        torch.nn.Tanh(),

        # 3rd layer (conv) 
        SelfONNLayer(in_channels=16,out_channels=16,kernel_size=5,stride=1,padding=3,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.Tanh(),

        # 4th layer (conv)
        SelfONNLayer(in_channels=16,out_channels=10,kernel_size=5,stride=1,padding=3,dilation=1,groups=1,bias=True,q=q_order,mode='fast',dropout=0.2),
        torch.nn.Tanh(), 
        torch.nn.AvgPool2d(3),

        # 5th layer (conv) 
        SelfONNLayer(in_channels=10,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.Tanh(),

        # 6th layer (conv) 
        SelfONNLayer(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast',dropout=0.2),
        torch.nn.Tanh(),
        torch.nn.AvgPool2d(3),

        # 7th layer (conv)
        SelfONNLayer(in_channels=16,out_channels=12,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast',dropout=0.2),
        torch.nn.MaxPool2d(3),
        torch.nn.Tanh(), 

        # 8th layer (conv)
        SelfONNLayer(in_channels=12,out_channels=250,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast',dropout=0.5),
        torch.nn.Tanh(), 
        torch.nn.AvgPool2d(3),
        # flatten 
        torch.nn.Flatten(),  
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model
 


def SelfONN_4(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.Tanh(), 
        # 3rd layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 4th layer (conv)
        SelfONNLayer(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(), 
        # 5th layer (conv) 
        SelfONNLayer(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 6th layer (conv)
        SelfONNLayer(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),  
        torch.nn.Tanh(), 
        # flatten 
        torch.nn.Flatten(),  
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model 


def SelfONN_5(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.Tanh(), 
        # 3rd layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 4th layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.Tanh(), 
        # 5th layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 6th layer (conv)
        SelfONNLayer(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.Tanh(), 
        # 7th layer (conv) 
        SelfONNLayer(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 8th layer (conv)
        SelfONNLayer(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(3),  
        torch.nn.Tanh(), 
        # flatten 
        torch.nn.Flatten(),  
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model 

def get_pretrained_model(parentdir, model_name, freeze_CNN, ImageNet,input_ch,class_num,q_order):
    """Retrieve a pre-trained model from torchvision

    Params
    -------
        model_name (str): name of the model (currently only accepts vgg16 and resnet50)

    Return
    --------
        model (PyTorch model): cnn

    """
  
    # size of flatten features for custom CNN models
    n_inputs = 1280
    if model_name == 'cnn_V1': 
        model = cnn_V1(input_ch,class_num)  
    elif model_name == 'cnn_V2': 
        model = cnn_V2(input_ch,class_num)  
    elif model_name == 'cnn_V3': 
        model = cnn_V3(input_ch,class_num)  
    elif model_name == 'cnn_V4': 
        model = cnn_V4(input_ch,class_num)  
    elif model_name == 'cnn_V5':   
        model = cnn_V5(input_ch,class_num)    
    elif model_name == 'SelfONN_1': 
        model = SelfONN_1(input_ch, class_num, q_order) 
    elif model_name == 'SelfONN_2': 
        model = SelfONN_2(input_ch, class_num, q_order) 
    elif model_name == 'SelfONN_3':  
        model = SelfONN_3(input_ch, class_num, q_order) 
    elif model_name == 'SelfONN_4': 
        model = SelfONN_4(input_ch, class_num, q_order) 
    elif model_name == 'SelfONN_5':  
        model = SelfONN_5(input_ch, class_num, q_order) 
    elif model_name == 'SelfONN_mArSL':
        model = SelfONN_mArSL(input_ch, class_num, q_order) 
    

    elif model_name == 'squeezenet1_0':
        from squeezenet import squeezenet1_0
        model = squeezenet1_0(pretrained=ImageNet) 
        model.classifier[-1] = nn.Sequential(  
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),  
            )   
        n_inputs = 1000
        
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=ImageNet)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        n_inputs = model.classifier[-1].in_features
        # Add on classifier
        model.classifier[-1] = nn.Identity()
        

    elif model_name == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=ImageNet) 
        n_inputs = model.classifier[-1].in_features
        # Add on classifier
        model.classifier[-1] = nn.Identity()

    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=ImageNet)
        n_inputs = model.classifier[-1].in_features
        # Add on classifier
        model.classifier[-1] = nn.Identity()

    elif model_name == 'vgg19_bn':
        model = models.vgg19_bn(pretrained=ImageNet)
        n_inputs = model.classifier[-1].in_features
        # Add on classifier
        model.classifier[-1] = nn.Identity()

    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=ImageNet)
        n_inputs = model.fc.in_features 
        model.fc = nn.Identity()

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=ImageNet)
        n_inputs = model.fc.in_features
        model.fc = nn.Identity()

    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=ImageNet)
        n_inputs = model.fc.in_features
        model.fc = nn.Identity()

    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=ImageNet)
        n_inputs = model.fc.in_features
        model.fc = nn.Identity()

    elif model_name == 'inception_v3':
        from Inception_Networks import inception_v3
        model = inception_v3(pretrained=ImageNet) 
        num_ftrs_Aux = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs_Aux, class_num) 
        # Handle the primary net
        n_inputs = model.fc.in_features 
        model.fc = nn.Identity()


    elif model_name == 'inception_v4':
        import pretrainedmodels
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        model = pretrainedmodels.__dict__['inceptionv4'](num_classes=1000, pretrained='imagenet')
        ssl._create_default_https_context = ssl._create_stdlib_context   
        n_inputs = model.last_linear.in_features 
        model.last_linear = nn.Identity()
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 


    elif model_name == 'inceptionresnetv2':
        from inceptionresnetv2 import inceptionresnetv2
        model = inceptionresnetv2(parentdir, num_classes=1000, pretrained='imagenet')
        n_inputs = model.last_linear.in_features
        model.last_linear = nn.Identity()
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 

    elif model_name == 'xception':
        from xception import xception
        model = xception(pretrained=ImageNet)
        n_inputs = model.fc.in_features
        model.fc = nn.Identity()

    elif model_name == 'chexnet':
        from chexnet import chexnet
        model = chexnet(parentdir) 
        n_inputs = model.module.densenet121.classifier[0].in_features
        model.module.densenet121.classifier = nn.Identity()

    elif model_name == 'nasnetalarge':
        import pretrainedmodels
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        ssl._create_default_https_context = ssl._create_stdlib_context 
        n_inputs = model.last_linear.in_features
        model.last_linear = nn.Identity()
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 

    elif model_name == 'pnasnet5large':
        import pretrainedmodels
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        ssl._create_default_https_context = ssl._create_stdlib_context 
        n_inputs = model.last_linear.in_features
        model.last_linear = nn.Identity() 
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 

    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=ImageNet) 
        n_inputs = model.classifier.in_features
        model.classifier = nn.Identity()

    elif model_name == 'densenet161':
        model = models.densenet161(pretrained=ImageNet)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier
        n_inputs = model.classifier.in_features
        model.classifier = nn.Identity()

    elif model_name == 'densenet201': 
        model = models.densenet201(pretrained=ImageNet)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier 
        n_inputs = model.classifier.in_features
        model.classifier = nn.Identity()

    elif model_name == 'shufflenet':
        model = models.shufflenet_v2_x1_0(pretrained=ImageNet)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=True)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier
        n_inputs = model.fc.in_features
        model.fc = nn.Identity()

    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=ImageNet)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=True)
        # Add on classifier
        n_inputs = model.fc.in_features
        model.fc = nn.Identity()

    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=ImageNet)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=True)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier 
        n_inputs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()

    elif model_name == 'nasnetamobile':
        import pretrainedmodels
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        ssl._create_default_https_context = ssl._create_stdlib_context 
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier
        n_inputs = model.last_linear.in_features
        model.last_linear = nn.Identity()

    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=ImageNet)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=True)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier 
        n_inputs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()

    elif model_name == 'darknet53':
        from darknet53 import darknet53 
        model = darknet53(1000)
        checkpoint = parentdir + 'models/darknet53.pth.tar'
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint ['state_dict'])
        del checkpoint 
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier 
        n_inputs = model.fc.in_features 
        model.fc = nn.Identity()
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 


    elif model_name == 'efficientnet_b0':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b0') 
        n_inputs = model._fc.in_features 
        model._fc = nn.Identity()
        model._swish =  nn.Identity() 
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 

    elif model_name == 'efficientnet_b1':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b1') 
        n_inputs = model._fc.in_features 
        model._fc = nn.Identity()
        model._swish =  nn.Identity() 
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 

    elif model_name == 'efficientnet_b2':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b2') 
        n_inputs = model._fc.in_features 
        model._fc = nn.Identity()
        model._swish =  nn.Identity() 
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 

    elif model_name == 'efficientnet_b3':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b3') 
        n_inputs = model._fc.in_features 
        model._fc = nn.Identity()
        model._swish =  nn.Identity() 
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 

    elif model_name == 'efficientnet_b4':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b4') 
        n_inputs = model._fc.in_features 
        model._fc = nn.Identity()
        model._swish =  nn.Identity() 
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 

    elif model_name == 'efficientnet_b5':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b5') 
        n_inputs = model._fc.in_features 
        model._fc = nn.Identity()
        model._swish =  nn.Identity() 
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 

    elif model_name == 'efficientnet_b6':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b6') 
        n_inputs = model._fc.in_features 
        model._fc = nn.Identity()
        model._swish =  nn.Identity() 
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 

    elif model_name == 'efficientnet_b7':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b7') 
        n_inputs = model._fc.in_features 
        model._fc = nn.Identity()
        model._swish =  nn.Identity() 
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 

    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained=True)
        n_inputs = model.classifier[3].in_features
        model.classifier[3] = nn.Identity()
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 
    
    elif model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(pretrained=True)
        n_inputs = model.classifier[3].in_features
        model.classifier[3] = nn.Identity()
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 


    elif model_name == 'convnext_tiny':
        model = models.convnext_tiny(pretrained=True)
        n_inputs = model.classifier[2].in_features
        model.classifier[2] = nn.Identity()
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 


    elif model_name == 'modified_mobileNet':
        mnet = models.mobilenet_v2(pretrained=True) 
        model = ModifiedMobileNet_V2(mnet)
        n_inputs=15680
        
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 

    if freeze_CNN:
            for param in model.parameters():
                param.requires_grad = False
        
    return model, n_inputs 


