import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

#%matplotlib inline

data = pd.read_csv("240EDITTED.csv")
data['Gender'] = data['Gender'].map({'F':0,'M':1}).astype(np.int)
data['Vomiting'] = data['Vomiting'].map({'N':0, 'Y':1}).astype(np.int)
data['Nausea'] = data['Nausea'].map({'N':0, 'Y':1}).astype(np.int)
data['Anorexia'] = data['Anorexia'].map({'N':0, 'Y':1}).astype(np.int)
data['migrationofPaintoRLQ'] = data['migrationofPaintoRLQ'].map({'N':0, 'Y':1}).astype(np.int)
data['abdominalcompressibility'] = data['abdominalcompressibility'].map({'N':0, 'Y':1}).astype(np.int)
data['RLQTendernessandReboundPain'] = data['RLQTendernessandReboundPain'].map({'N':0, 'Y':1}).astype(np.int)
data['Fever'] = data['Fever'].map({'N':0, 'Y':1}).astype(np.int)
data['Cough'] = data['Cough'].map({'N':0, 'Y':1}).astype(np.int)
data['Diarrhea'] = data['Diarrhea'].map({'N':0, 'Y':1}).astype(np.int)

data.dtypes

data.isnull().values.any()

Prediction_map = {True: 1, False: 0}
data['Prediction'] = data['Prediction'].map(Prediction_map)
data.head(5)
Prediction_true_count = len(data.loc[data['Prediction'] == True])
Prediction_false_count = len(data.loc[data['Prediction'] == False])

from sklearn.model_selection import train_test_split
feature_columns = ['Age', 'Gender', 'Vomiting', 'Nausea', 'Anorexia', 'migrationofPaintoRLQ', 'RLQTendernessandReboundPain','Tempraturec','DuirationofsymptomHours','Fever','Cough','Diarrhea','HCT','WBC','PLT','NEUT','MCHC','RBC','HGB','lymph']
predicted_class = ['Prediction']

X = data[feature_columns].values
y = data[predicted_class].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=10)

from sklearn.impute import SimpleImputer

fill_values = SimpleImputer(missing_values=0, strategy="mean")

X_train = fill_values.fit_transform(X_train)
X_test = fill_values.fit_transform(X_test)

from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state=10)

random_forest_model.fit(X_train, y_train.ravel())

predict_train_data = random_forest_model.predict(X_test)

from sklearn import metrics

print("Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))
print(random_forest_model.predict([['16','1','1','0','1','1','0','35.4','198','1','0','0','41','9.6','159','45','30.2','3','10.9','2.30']]))

pickle.dump(random_forest_model,open('good.pkl','wb'))