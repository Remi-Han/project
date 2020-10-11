import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt
col_names = ['winner','firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron',
             'firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills',
             't1_baronKills','t1_dragonKills','t1_riftHeraldKills',
             't2_towerKills','t2_inhibitorKills','t2_baronKills',
             't2_dragonKills','t2_dragonKills']
# load data
mod = pd.read_csv('new_data.csv')#training set
test = pd.read_csv('test_set.csv')#testing set
#remap
mappings = {
    1:0,
    2:1
}
mod['winner'] = mod['winner'].apply(lambda x: mappings[x])
test['winner'] = test['winner'].apply(lambda x: mappings[x])
#split the data
feature_cols = ['firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron',
                'firstDragon','firstRiftHerald','t1_towerKills',
                't1_inhibitorKills','t1_baronKills','t1_dragonKills',
                't1_riftHeraldKills','t2_towerKills','t2_inhibitorKills',
                't2_baronKills','t2_dragonKills','t2_dragonKills'] 

X_train = mod[feature_cols].values # features
y_train = mod.winner.values # target value
X_test = test[feature_cols].values
y_test = test.winner.values 
#use torch 
X_train = torch.FloatTensor(X_train) 
X_test = torch.FloatTensor(X_test) 
y_train = torch.LongTensor(y_train) 
y_test = torch.LongTensor(y_test)
#ANN model
class ANN(nn.Module): 
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=16, out_features=100) 
        self.output = nn.Linear(in_features=100, out_features=2)
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x)) 
        x = self.output(x)
        x = F.softmax(x,dim=1)
        return x    
model = ANN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#train the model
epochs = 1000
loss_arr = []
for i in range(epochs):
    y_hat = model.forward(X_train) 
    loss = criterion(y_hat, y_train) 
    loss_arr.append(loss)
    
    if i % 100 == 0:
        print(f'Epoch: {i} Loss: {loss}')
    
    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step()
    
#predict and calculate the accuracy 
predict_out = model(X_test)
_,predict_y = torch.max(predict_out, 1)
print("The accuracy is ", accuracy_score(y_test, predict_y) )
