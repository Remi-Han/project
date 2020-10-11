import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 

col_names = ['winner','gameId','creationTime', 'gameDuration', 'seasonId', 
             'firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron',
             'firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills',
             't1_baronKills','t1_dragonKills','t1_riftHeraldKills',
             't2_towerKills','t2_inhibitorKills','t2_baronKills',
             't2_dragonKills','t2_dragonKills']
# load dataset
mod = pd.read_csv('new_data.csv')#training set
mod = mod.iloc[1:] # delete the first row of the dataframe ts.head()
test = pd.read_csv('test_set.csv')#testing set
test = test.iloc[1:] # delete the first row of the dataframe ts.head()
#split dataset in features and target variable
feature_cols = ['gameId','creationTime', 'gameDuration','seasonId', 
                'firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron',
                'firstDragon','firstRiftHerald','t1_towerKills',
                't1_inhibitorKills','t1_baronKills','t1_dragonKills',
                't1_riftHeraldKills','t2_towerKills','t2_inhibitorKills',
                't2_baronKills','t2_dragonKills','t2_dragonKills'] 

X_train = mod[feature_cols] # Features
y_train = mod.winner # Target variable
X_test = test[feature_cols] # Features
y_test = test.winner # Target variable


# Create Decision Tree classifer object 
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer 
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset 
y_pred = clf.predict(X_test)

# Model's Accuracy 
print("The accuracy is ",accuracy_score(y_test, y_pred)) 

