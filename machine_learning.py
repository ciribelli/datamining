import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import pathlib
import joblib

path = pathlib.Path(__file__).parent.resolve()
df = pd.read_csv(pathlib.PurePath(path, 'datasetDM.csv'))
df=df.replace(to_replace="ON",value=1)
df=df.replace(to_replace="OFF",value=0)
df = df.loc[df['local_id'] == 1]
# features selecionadas
df = df.loc[:,['temp_media_15',	'temp_desvpad_60', 'temp_variancia_30',	'temp_media_30', 'temp_amplitude_60', 'classificacao']]
X = df.loc[:,df.columns != 'classificacao']  # Entrada
y = df.classificacao   # Saida
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)


############ NORMALIZACAO

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)



from sklearn.ensemble import RandomForestClassifier
def train(X_train, y_train):
    model = RandomForestClassifier(min_samples_leaf=5, bootstrap=False) # tente mudar parâmetro para evitar overfitting
    model.fit(X_train, y_train)
    return model

# from sklearn.ensemble import GradientBoostingClassifier
# def train(X_train, y_train):
#     model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0)
#     model.fit(X_train, y_train)
#     return model

# from sklearn.neural_network import MLPClassifier
# def train(X_train, y_train):
#     model = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001, solver='adam', verbose=True, random_state=21 ,tol=0.000001)
#     model.fit(X_train, y_train)
#     return model

# from sklearn import svm
# def train(X_train, y_train):
#     model = svm.SVC()
#     model.fit(X_train, y_train)
#     return model



model = train(X_train, y_train)
joblib.dump(model, 'ml.pkl', compress=9) # salva o arquivo pickle

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

