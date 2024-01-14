import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# laod data
file_path = 'C:\\Users\\diogo\\OneDrive - Instituto Politécnico do Cávado e do Ave\\Desktop\\ML_TP\\DataSet.xlsx'
data = pd.read_excel(file_path)

print(data.shape)
print(data.head(5))

# Removed because it is a id column
data = data.drop(['Unnamed: 0'], axis=1)
print(data.shape)

# # split data into X and Y
# X = data.drop(['Abandono'], axis = 1)  # Input_set  
# Y = data['Abandono'] # Output

# # split data into train and test sets
# test_size = 0.2
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, shuffle = True)

# print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


class_0 = data[data['Abandono'] == 0]
class_1 = data[data['Abandono'] == 1]

class_0_sample = class_0.sample(600, random_state=42)  
class_1_sample = class_1.sample(600, random_state=42)  

data_balanced = pd.concat([class_0_sample, class_1_sample])
print(data_balanced.shape)

data_balanced = data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

X_balanced = data_balanced.drop('Abandono', axis=1)
y_balanced = data_balanced['Abandono']


X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

correlation_matrix = data_balanced.corr()

# And then use seaborn's heatmap to visualize it
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, cmap='BrBG', fmt='.2f', linewidths=2, annot=True)
plt.title('Matriz de Correlação')
plt.show()


from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, accuracy,precision,recall,f1, title='Confusion Matrix'):
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(matrix, annot=True, fmt="d", linewidths=.5, cmap="Blues")
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predict')
    
    metrics_string = (f'Accuracy: {accuracy:.4f} | '
                      f'Precision: {precision:.4f} | '
                      f'Recall: {recall:.4f} | '
                      f'F1-Score: {f1:.4f}')
    
    # Define a posição do texto, por exemplo, x será 0.5 (centro do gráfico)
    # e y será -0.1 (abaixo do eixo x do gráfico).
    # Ajuste esses valores conforme necessário para a posição correta.
    plt.text(0.5, -0.1, metrics_string, ha='center', va='top', transform=plt.gca().transAxes)
    
    plt.show()


#### Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_validate

rfc = RandomForestClassifier( min_samples_split=2, min_samples_leaf=2, max_depth=10)
rfc.fit(X_train, y_train)
predictions = rfc.predict(X_test)
print(classification_report(y_test, predictions))
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='binary')
recall = recall_score(y_test, predictions, average='binary')
f1 = f1_score(y_test, predictions, average='binary')

cv_results = cross_validate(rfc, X_train, y_train, cv=5, return_train_score=True, scoring='accuracy')
cv_train_scores = cv_results['train_score']
cv_test_scores = cv_results['test_score']
print("Train scores: ", cv_train_scores)
print("Test scores: ", cv_test_scores)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')

plot_confusion_matrix(y_test, predictions,accuracy,precision,recall,f1, title='Confusion Matrix - Random Forest')


dtc = DecisionTreeClassifier(splitter='best', min_samples_split=7, min_samples_leaf=3, max_depth=5, criterion='gini')
dtc.fit(X_train, y_train)
predictions = dtc.predict(X_test)
print(classification_report(y_test, predictions))
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='binary')
recall = recall_score(y_test, predictions, average='binary')
f1 = f1_score(y_test, predictions, average='binary')

cv_results = cross_validate(dtc, X_train, y_train, cv=5, return_train_score=True, scoring='accuracy')
cv_train_scores = cv_results['train_score']
cv_test_scores = cv_results['test_score']
print("Train scores: ", cv_train_scores)
print("Test scores: ", cv_test_scores)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')

plot_confusion_matrix(y_test, predictions,accuracy,precision,recall,f1, title='Confusion Matrix - Decision Tree')

lrc = LogisticRegression()
lrc.fit(X_train, y_train)
predictions = lrc.predict(X_test)
print(classification_report(y_test, predictions))
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='binary')
recall = recall_score(y_test, predictions, average='binary')
f1 = f1_score(y_test, predictions, average='binary')

cv_results = cross_validate(lrc, X_train, y_train, cv=5, return_train_score=True, scoring='accuracy')
cv_train_scores = cv_results['train_score']
cv_test_scores = cv_results['test_score']
print("Train scores: ", cv_train_scores)
print("Test scores: ", cv_test_scores)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')

plot_confusion_matrix(y_test, predictions,accuracy,precision,recall,f1, title='Confusion Matrix - Logistic Regression')



from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint
from scipy.stats import uniform

from sklearn.model_selection import GridSearchCV

# # Defina os hiperparâmetros a serem testados para o DecisionTreeClassifier
# param_grid = {
#     'criterion': ['gini', 'entropy'],  # Critério para medir a qualidade da divisão
#     'splitter': ['best', 'random'],    # Estratégia de divisão
#     'max_depth': [None, 5, 10, 20],   # Profundidade máxima da árvore
#     'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9],  # Número mínimo de amostras para dividir um nó interno
#     'min_samples_leaf': [1, 2, 3, 4]     # Número mínimo de amostras por folha
# }

# # Crie o objeto GridSearchCV para o DecisionTreeClassifier
# grid_search_dt = RandomizedSearchCV(estimator=DecisionTreeClassifier(), param_distributions=param_grid, cv=5)

# # Realize a busca em grade
# grid_search_dt.fit(X_train, y_train)

# # Imprima os melhores hiperparâmetros encontrados
# print("Melhores hiperparâmetros para o modelo DecisionTreeClassifier:")
# print(grid_search_dt.best_params_)



# # Defina os hiperparâmetros a serem testados para a Regressão Logística
# param_grid = {
#     'C': [0.1, 1, 10],  # Parâmetro de regularização
#     'penalty': ['l1', 'l2'],  # Tipo de penalização (L1 ou L2)
#     'solver': ['liblinear', 'saga'],  # Algoritmo de otimização
#     'max_iter': [100, 200, 300, 400, 500]  # Número máximo de iterações
# }

# # Crie o objeto GridSearchCV para a Regressão Logística
# grid_search_lr = RandomizedSearchCV(estimator=LogisticRegression(), param_distributions=param_grid, cv=5)

# # Realize a busca em grade
# grid_search_lr.fit(X_train, y_train)

# # Imprima os melhores hiperparâmetros encontrados
# print("Melhores hiperparâmetros para o modelo de Regressão Logística:")
# print(grid_search_lr.best_params_)



# # Definir os hiperparâmetros a serem testados
# param_grid = {
#     'n_estimators': [100, 200, 300, 400, 500],
#     'max_depth': [None, 5, 10, 20],
#     'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9],
#     'min_samples_leaf': [1, 2, 3, 4]
# }

# # Criar o objeto GridSearchCV
# grid_search_rf = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=param_grid, cv=5)

# # Realizar a busca em grade
# grid_search_rf.fit(X_train, y_train)

# # Imprimir os melhores hiperparâmetros encontrados
# print("Melhores hiperparâmetros para o modelo RandomForestClassifier:")
# print(grid_search_rf.best_params_)