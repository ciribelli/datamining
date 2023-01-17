import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pathlib
import joblib
from sklearn.metrics import classification_report



# Inicio

path = pathlib.Path(__file__).parent.resolve()
df = pd.read_csv(pathlib.PurePath(path, 'datasetDM.csv'))
df = df.loc[df['local_id'] == 0] # para fazer validacoes adicionais com a classe do local_id = 0

model = joblib.load(pathlib.PurePath(path, 'ml.pkl'))

x = df.data
temperatura = df.temperatura
y = df.classificacao   # Saida
X = df.loc[:,['temp_media_15',	'temp_desvpad_60', 'temp_variancia_30',	'temp_media_30', 'temp_amplitude_60']]
y_pred = model.predict(X) # inferência do teste

print(classification_report(y, y_pred))

# ajuste da janela de plot para efeito de otimizacao do tempo
d = 1500
i = 0 + d
f = 1000 + d

fig, ax = plt.subplots()
ax.plot(x[i:f], y[i:f], '--', label = 'real')
ax.plot(x[i:f], temperatura[i:f]/15, '-', color = 'red', label = 'temperatura')
ax.fill_between(x[i:f], y[i:f], 0, alpha=0.2)
ax.plot(x[i:f], y_pred[i:f], '-', color='green', label = 'previsto')
plt.legend(loc="lower left")


plt.show()
