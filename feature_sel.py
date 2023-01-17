import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#df = pd.read_csv('/Users/ciribelli/Server/Data Mining/datasetDM.csv')
df = pd.read_csv('C://Users//Ciribelli//OneDrive//Documentos//BI Master//DM//datamining//datasetDM.csv')

df = df.loc[df['local_id'] == 1]

# opcoes de janela para selecao de parametros
#df = df.loc[:,['classificacao', 'umid_amplitude_30',	'umid_desvpad_30',	'umid_media_30', 'umid_variancia_30']] # local_id = 0
#df = df.loc[:,['temp_desvpad_60','temp_variancia_60', 'temp_media_60', 'temp_amplitude_60', 'classificacao']] # local_id = 1
#df = df.loc[:,['temp_desvpad_30', 'temp_variancia_30',  'temp_media_30', 'temp_amplitude_30', 'classificacao']] # local_id = 1

# opcao selecionada
df = df.loc[:,['temp_media_15',	'temp_desvpad_60', 'temp_variancia_30',	'temp_media_30', 'temp_amplitude_60', 'classificacao']] # local_id = 1

print(df.head())

sns.pairplot(df,diag_kind="kde",hue="classificacao",palette="husl")
plt.show()




