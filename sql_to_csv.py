import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import pathlib


def load_sql(path):
    con = sqlite3.connect(pathlib.PurePath(path, 'db.sqlite3'))
    saida = pd.read_sql_query("SELECT * from controleambiente_ambiente", con)
    saida['data'] =  pd.to_datetime(saida['data'], format='%Y-%m-%d %H:%M:%S')
    con.close()
    return saida

def filtra(s):
    to_remove = s[s['temperatura'] < 15.0].index
    s = s.drop(to_remove)
    to_remove = s[s['umidade'] < 30.0].index
    s = s.drop(to_remove)
    return s

def plota(f, s, item, a):
    axs[item, 0].plot(s.index, s['umidade'], f.index, f['umidade'])
    axs[item, 1].plot(s.index, s['temperatura'], f.index, f['temperatura'])
    axs[item, 0].grid()
    axs[item, 1].grid()
    return(plt)

def classifica(df, i):
    dfs = df.set_index('data')[i['t0']:i['tf']]
    dfs = dfs.loc[dfs['local_id'] == i['local_id']]
    dfs = dfs.drop(['id'], axis=1)
    dfs['classificacao'] = 1
    dfs['classificacao'] = np.where(dfs.index >= i['toff'], 0, dfs['classificacao'])
    return dfs

def segmentaInicio(u):
    u = u.loc[i['ti']:i['tf']]
    return u

def aplica_janela(daux):

    janela_0 = 15 # janela deslizante de 15 elementos
    janela_1 = 30 # janela deslizante de 30 elementos
    janela_2 = 60 # janela deslizante de 60 elementos

    l_janelas = [janela_0, janela_1, janela_2]

    for l_j in l_janelas:

        # para temperaturas
        s_arcon_t = daux['temperatura'].rolling(l_j).std()              # desvio padrao
        v_arcon_t = daux['temperatura'].rolling(l_j).var()              # variância
        m_arcon_t = daux['temperatura'].rolling(l_j).mean()             # média
        min_arcon_t = daux['temperatura'].rolling(l_j).min()            # calc. auxiliar de mínimo
        max_arcon_t = daux['temperatura'].rolling(l_j).max()            # calc. auxiliar de máximo
        a_arcon_t = max_arcon_t - min_arcon_t           # amplitude (max - min)

        # para umidade
        s_arcon_u = daux['umidade'].rolling(l_j).std()              # desvio padrao
        v_arcon_u = daux['umidade'].rolling(l_j).var()              # variância
        m_arcon_u = daux['umidade'].rolling(l_j).mean()             # média
        min_arcon_u = daux['umidade'].rolling(l_j).min()            # calc. auxiliar de mínimo
        max_arcon_u = daux['umidade'].rolling(l_j).max()            # calc. auxiliar de máximo
        a_arcon_u = max_arcon_u - min_arcon_u           # amplitude (max - min)

        daux['temp_desvpad_' + str(l_j)] = s_arcon_t
        daux['temp_variancia_' + str(l_j)] = v_arcon_t
        daux['temp_media_' + str(l_j)] = m_arcon_t
        daux['temp_amplitude_' + str(l_j)] = a_arcon_t
        daux['umid_desvpad_' + str(l_j)] = s_arcon_u
        daux['umid_variancia_' + str(l_j)] = v_arcon_u
        daux['umid_media_' + str(l_j)] = m_arcon_u
        daux['umid_amplitude_' + str(l_j)] = a_arcon_u
    return (daux)


# Inicio
path = pathlib.Path(__file__).parent.resolve()
df = load_sql(path) # dataframe original
ds = filtra(df) # dataframe filtrado

intervalo_01 = {'t0': '2018-12-05 15:10', 'ti': '2018-12-05 16:28', 'tf': '2018-12-06 14:00', 'toff': '2018-12-06 01:36', 'local_id': 0}
intervalo_02 = {'t0': '2018-10-03 20:30', 'ti': '2018-10-03 21:43', 'tf': '2018-10-04 22:40', 'toff': '2018-10-04 02:43', 'local_id': 0}
intervalo_03 = {'t0': '2018-11-28 15:10', 'ti': '2018-11-28 16:23', 'tf': '2018-11-29 22:40', 'toff': '2018-11-29 01:07', 'local_id': 0}
intervalo_04 = {'t0': '2020-04-07 22:15', 'ti': '2020-04-07 23:19', 'tf': '2020-04-08 17:48', 'toff': '2020-04-08 10:30', 'local_id': 1}
intervalo_05 = {'t0': '2020-04-12 22:25', 'ti': '2020-04-12 23:30', 'tf': '2020-04-13 22:00', 'toff': '2020-04-13 10:46', 'local_id': 1}
intervalo_06 = {'t0': '2020-04-11 00:05', 'ti': '2020-04-11 01:12', 'tf': '2020-04-11 16:52', 'toff': '2020-04-11 11:32', 'local_id': 1}

l = [intervalo_01, intervalo_02, intervalo_03, intervalo_04, intervalo_05, intervalo_06]

fig, axs = plt.subplots(6, 2)
size=8
params = {'legend.fontsize': 'large',
          'figure.figsize': (20,8),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.75,
          'ytick.labelsize': size*0.75,
          'axes.titlepad': 25}
plt.rcParams.update(params)
dF = pd.DataFrame()

# item eh o indice do for para geracao dos graficos // i eh o objeto intervalo
for item, i in enumerate(l):
    # segmenta e adiciona classificacao
    dff = classifica(df, i) # c/ o original
    dfs = classifica(ds, i) # c/ o filtrado
    # aplica janela
    dfa = aplica_janela(dfs)
    # ajusta inicio do segmento de t0 para ti
    dfa = segmentaInicio(dfa)
    # plota
    plota(dfa, dff, item, axs)
    # concatena
    dF = pd.concat([dF, dfa])

plt.show()

# salva arquivo csv
dF.to_csv('C://Users//Ciribelli//OneDrive//Documentos//BI Master//DM//datamining//datasetDM.csv', index=True)
print('Arquivo salvo')





