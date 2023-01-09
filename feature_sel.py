import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import seaborn as sns

df = pd.read_csv('/Users/ciribelli/Server/Data Mining/datasetDM.csv')


df = df.loc[df['local_id'] == 0]

# 3 opcoes de janela para selecao de parametros
#df = df.loc[:,['temp_desvpad_15',	'temp_variancia_15',	'temp_media_15',	'temp_amplitude_15', 'classificacao']]#, 'temp_desvpad_30',	'temp_variancia_30',	'temp_media_30',	'temp_amplitude_30',	'temp_desvpad_60',	'temp_variancia_60',	'temp_media_60',	'temp_amplitude_60']]
#df = df.loc[:,['temp_desvpad_30',	'temp_variancia_30',	'temp_media_30',	'temp_amplitude_30', 'classificacao']]
#df = df.loc[:,['temp_desvpad_60',	'temp_variancia_60',	'temp_media_60',	'temp_amplitude_60', 'classificacao']]
df = df.loc[:,['classificacao', 'umid_amplitude_30',	'umid_desvpad_30',	'umid_media_30', 'umid_variancia_30', 'umidade']]

print(df.head())

sns.pairplot(df,diag_kind="kde",hue="classificacao",palette="husl")
plt.show()




