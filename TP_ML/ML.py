import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# laod data
file_path = 'C:\\Users\\diogo\\OneDrive - Instituto Politécnico do Cávado e do Ave\\Desktop\\ML_TP\\DataSet.xlsx'
data = pd.read_excel(file_path)

print(data.shape)

# Removed because it is a index column
data = data.drop(['Unnamed: 0'], axis=1)

correlation_matrix = data.corr()

# And then use seaborn's heatmap to visualize it
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, cmap='BrBG', fmt='.2f', linewidths=2, annot=True)
plt.title('Matriz de Correlação')
plt.show()



from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# Separe os dados em duas classes
data_class_0 = data[data['Abandono'] == 0]
data_class_1 = data[data['Abandono'] == 1]

# Faça o undersampling da classe majoritária
data_class_0_under = resample(data_class_0, 
                              replace=False, 
                              n_samples=len(data_class_1), 
                              random_state=42)

# Combine as classes novamente
data_balanced = pd.concat([data_class_0_under, data_class_1])

# Agora dividimos em conjuntos de treino e teste
X = data_balanced.drop('Abandono', axis=1)
Y = data_balanced['Abandono']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, shuffle = True, stratify=Y)


# # split data into X and Y
# X = data.drop(['Abandono'], axis = 1)  # Input_set  
# Y = data['Abandono'] # Output

# # split data into train and test sets
# test_size = 0.3
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, shuffle = True)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)



#### Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

dtc= DecisionTreeClassifier()
dtc.fit(X_train.values, Y_train)
dtc_predictions= dtc.predict(X_test.values)

# Example
dtc_predictions_example = dtc.predict([[1,	1,	1,	1,	1,	1,	0,	26,	0,	118,	1,	0,	1,	1,	0,	0,	0,	0,	0,	0,	1,	1,	0,	0,	0,	1,	1,	1,	0,	0,	0,	0], 
                                       [0,	11,	1,	1,	0,	1,	0,	28,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	1,	1,	1,	0,	0,	1,	0,	0,	1,	0,	1,	0,	0,	0]])
print(dtc_predictions_example)
# 0-Não Abandona; 1-Abandona;



# Accuracy
from sklearn.metrics import accuracy_score

score_DTC=accuracy_score(Y_test, dtc_predictions)
print("Accuracy of Decision Tree: %.2f%%" % (score_DTC * 100.0)) # This score is always different since the division between train and test is always random
# 80% - 85% Accuracy



#### XGBoost Classifier
from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train.values, Y_train)
xgb_predictions = xgb.predict(X_test.values)
score_XGB = accuracy_score(Y_test, xgb_predictions)
print("Accuracy of XGBoost: %.2f%%" % (score_XGB * 100.0)) # This score is always different since the division between train and test is always random



#### Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression

lrc = LogisticRegression(max_iter = 10000)
lrc.fit(X_train.values, Y_train)
lrc_predictions = lrc.predict(X_test.values)
score_LRC = accuracy_score(Y_test, lrc_predictions)
print("Accuracy of Logistic Regression: %.2f%%" % (score_LRC * 100.0)) # This score is always different since the division between train and test is always random



#### Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(X_train.values, Y_train)
rfc_predictions = rfc.predict(X_test.values)
score_RFC = accuracy_score(Y_test, rfc_predictions)
print("Accuracy of Random Forest: %.2f%%" % (score_RFC * 100.0)) # This score is always different since the division between train and test is always random



from sklearn.metrics import confusion_matrix

        ####### Confusion Matrix  ########
# Logistic Regression
cm = confusion_matrix(Y_test, lrc_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# XGBoost
cm = confusion_matrix(Y_test, xgb_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# Random Forest
cm = confusion_matrix(Y_test, rfc_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# Decision Tree
cm = confusion_matrix(Y_test, dtc_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()



from sklearn.metrics import precision_score, recall_score, f1_score

# Calcular precisão, recall e F1-score para os 4 modelos
print('\n\nLOGISTIC REGRESSION')
lrc_precision = precision_score(Y_test, lrc_predictions, average='binary')
lrc_recall = recall_score(Y_test, lrc_predictions, average='binary')
lrc_f1 = f1_score(Y_test, lrc_predictions, average='binary')

print(f'Precision: {lrc_precision:.2f}')
print(f'Recall: {lrc_recall:.2f}')
print(f'F1 Score: {lrc_f1:.2f}')



print('\n\nDECISION TREE CLASSIFIER')
dtc_precision = precision_score(Y_test, dtc_predictions, average='binary')
dtc_recall = recall_score(Y_test, dtc_predictions, average='binary')
dtc_f1 = f1_score(Y_test, dtc_predictions, average='binary')

print(f'Precision: {dtc_precision:.2f}')
print(f'Recall: {dtc_recall:.2f}')
print(f'F1 Score: {dtc_f1:.2f}')



print('\n\nRANDOM FOREST CLASSIFIER')
rfc_precision = precision_score(Y_test, rfc_predictions, average='binary')
rfc_recall = recall_score(Y_test, rfc_predictions, average='binary')
rfc_f1 = f1_score(Y_test, rfc_predictions, average='binary')

print(f'Precision: {rfc_precision:.2f}')
print(f'Recall: {rfc_recall:.2f}')
print(f'F1 Score: {rfc_f1:.2f}')



print('\n\nXGBOOST CLASSIFIER')
xgb_precision = precision_score(Y_test, xgb_predictions, average='binary')
xgb_recall = recall_score(Y_test, xgb_predictions, average='binary')
xgb_f1 = f1_score(Y_test, xgb_predictions, average='binary')

print(f'Precision: {xgb_precision:.2f}')
print(f'Recall: {xgb_recall:.2f}')
print(f'F1 Score: {xgb_f1:.2f}')

