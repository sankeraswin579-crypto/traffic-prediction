# Traffic Prediction using Time Series

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("Sample ML Project Running")

# dummy dataset
data = pd.DataFrame({
    'feature1':[1,2,3,4,5],
    'feature2':[5,4,3,2,1],
    'target':[0,1,0,1,0]
})

X = data[['feature1','feature2']]
y = data['target']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train,y_train)

print("Model trained successfully")
