import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np

df = pd.read_csv('C://Users//Ciribelli//OneDrive//Documentos//BI Master//DM//datamining//datasetDM.csv')
df=df.replace(to_replace="ON",value=1)
df=df.replace(to_replace="OFF",value=0)
df = df.loc[df['local_id'] == 1]
# features selecionadas
df = df.loc[:,['temp_media_15',	'temp_desvpad_60', 'temp_variancia_30',	'temp_media_30', 'temp_amplitude_60', 'classificacao']]
#df = df.loc[:,['classificacao', 'umid_amplitude_30',	'umid_desvpad_30',	'umid_media_30', 'umid_variancia_30']]
#df = df.loc[:,['temp_media_30',	'temp_desvpad_30', 'temp_variancia_30',	'temp_media_30', 'temp_amplitude_60', 'temperatura', 'classificacao']]
X = df.loc[:,df.columns != 'classificacao']  # Entrada
y = df.classificacao    # Saida
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)





############ FEATURE SELECTION

# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# X_new = SelectKBest(chi2, k=3).fit_transform(X, y)
# print(X_new)

############ NORMALIZACAO

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)




# treinar modelo
from sklearn.ensemble import RandomForestClassifier
def train(X_train, y_train):
    model = RandomForestClassifier(min_samples_leaf=5, bootstrap=False) # tente mudar parâmetro para evitar overfitting
    model.fit(X_train, y_train)
    return model

# treinar modelo
# from sklearn.ensemble import GradientBoostingClassifier
# def train(X_train, y_train):
#     model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0)
#     model.fit(X_train, y_train)
#     return model

#treinar modelo
# from sklearn.neural_network import MLPClassifier
# def train(X_train, y_train):
#     model = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001, solver='adam', verbose=True, random_state=21 ,tol=0.000001)
#     model.fit(X_train, y_train)
#     return model

from sklearn import svm
def train(X_train, y_train):
    model = svm.SVC()
    model.fit(X_train, y_train)
    return model



#### COMITE ####################

# from sklearn import datasets
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import VotingClassifier

# clf1 = LogisticRegression(random_state=1)
# clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
# clf3 = GaussianNB()

# eclf = VotingClassifier( estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='hard')
# for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']): 
#     scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=5)
# print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))




model = train(X_train, y_train)

def predict_and_evaluate(model, X_test, y_test):

    y_pred = model.predict(X_test) # inferência do teste

    # Acurácia
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    print('Acurácia: ', accuracy)

    # Kappa
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(y_test, y_pred)
    print('Kappa: ', kappa)

    # F1
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred)
    print('F1: ', f1)

    # Matriz de confusão
    # .: true negatives is C00, false negatives is C10, true positives C11 is  and false positives is C01.
    from sklearn.metrics import confusion_matrix
    confMatrix = confusion_matrix(y_test, y_pred)

    ax = plt.subplot()
    sns.heatmap(confMatrix, annot=True, fmt=".0f")
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')

    # Colocar os nomes
    ax.xaxis.set_ticklabels(['OFF', 'ON']) 
    ax.yaxis.set_ticklabels(['OFF', 'ON'])
    plt.show()

    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))

print('Resultados de Treino')
predict_and_evaluate(model, X_train, y_train)
print('Resultados de Teste')
predict_and_evaluate(model, X_test, y_test)


