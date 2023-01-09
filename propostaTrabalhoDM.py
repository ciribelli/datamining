import streamlit as st
import pandas as pd
import datetime
import numpy as np
import altair as alt
import pydeck as pdk
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import joblib


def filtra(s):
    k = s['temperatura']
    removed = k.between(k.quantile(.002), k.quantile(.998))
    filtrado = s[~removed].index # INVERT removed_outliers!!
    nova = s.drop(filtrado)
    l = nova['umidade']
    removed = l.between(l.quantile(.002), l.quantile(.998))
    #removed.size
    filtrado = nova[~removed].index # INVERT removed_outliers!!
    df = nova.drop(filtrado)
    return df

def plota(df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df['tempo'], y = df['umidade'], mode='lines', name='Umid', line={'color': 'blue'}),secondary_y=True)
    fig.add_trace(go.Scatter(x=df['tempo'], y = df['temperatura'], mode='lines', name='Temp'),secondary_y=False)
    fig.add_trace(go.Scatter(x=df['tempo'], y = df['classificacao']*10, mode='lines', name='ON/OFF'),secondary_y=True)
    fig.update_layout(
        yaxis_title="T[oC]",
        yaxis2_title="U[%]",
    )
    st.plotly_chart(fig)
    

def aplica_janela(saida):
    Y_t = saida['temperatura']
    Y_u = saida['umidade']
    T = saida['data']

    janela_0 = 15 # janela deslizante de 15 elementos
    janela_1 = 30 # janela deslizante de 30 elementos
    janela_2 = 60 # janela deslizante de 60 elementos

    l_janelas = [janela_0, janela_1, janela_2]

    df = pd.DataFrame({'tempo': T, 'temperatura': Y_t, 'umidade': Y_u})

    for l_j in l_janelas:

        # para temperaturas
        s_arcon_t = Y_t.rolling(l_j).std()              # desvio padrao
        v_arcon_t = Y_t.rolling(l_j).var()              # variância
        m_arcon_t = Y_t.rolling(l_j).mean()             # média
        min_arcon_t = Y_t.rolling(l_j).min()            # calc. auxiliar de mínimo
        max_arcon_t = Y_t.rolling(l_j).max()            # calc. auxiliar de máximo
        a_arcon_t = max_arcon_t - min_arcon_t           # amplitude (max - min)

        # para umidade
        s_arcon_u = Y_t.rolling(l_j).std()              # desvio padrao
        v_arcon_u = Y_t.rolling(l_j).var()              # variância
        m_arcon_u = Y_t.rolling(l_j).mean()             # média
        min_arcon_u = Y_t.rolling(l_j).min()            # calc. auxiliar de mínimo
        max_arcon_u = Y_t.rolling(l_j).max()            # calc. auxiliar de máximo
        a_arcon_u = max_arcon_u - min_arcon_u           # amplitude (max - min)

        df['temp_desvpad_' + str(l_j)] = s_arcon_t
        df['temp_variancia_' + str(l_j)] = v_arcon_t
        df['temp_media_' + str(l_j)] = m_arcon_t
        df['temp_amplitude_' + str(l_j)] = a_arcon_t
        df['umid_desvpad_' + str(l_j)] = s_arcon_u
        df['umid_variancia_' + str(l_j)] = v_arcon_u
        df['umid_media_' + str(l_j)] = m_arcon_u
        df['umid_amplitude_' + str(l_j)] = a_arcon_u
    return (df)


st.title("Trabalho Final Data Mining")
st.subheader("Proposta de trabalho para avaliação da Prof. Manoela Kohler")
st.subheader("Aluno: Otávio Ciribelli Borges \n email: otavio.ciribelli@gmail.com || tel: (21) 983163900")

################
st.subheader("Introdução")
st.write("""A proposta de trabalho consiste em implementar uma rotina de classificação do tipo supervisionada em um sistema de medição de condições climáticas em dois ambientes domésticos. 
As medidas de temperatura e umidade foram tomadas em períodos compreendidos entre 2018 e 2021 e estão disponíveis em repositório do github no 
link https://github.com/ciribelli/autohome. O arquivo em específico fica em: https://github.com/ciribelli/autohome/tree/master/home/db.sqlite3.
""")

st.write("""O objetivo de classificação deste trabalho consiste em prever, a partir dos registros de temperatura e umidade, quando
o aparelho de ar condicionado dos ambientes esteve ligado. Existe portanto duas classes de saída que são de natureza numérica, sendo '1' para o
estado de ar condicionado ligado e '0' quando o aparelho está desligado.""")

st.write("""Em termos de propósito e abrangência mercadológica, esta rotina de classificação tem potencial contribuição
com a simplificação de aplicações do tipo IoT ou IIoT, podendo atender com a geração de informações adicionais àquelas dos
sensores de variáveis físicas.""")
st.write("""-----""")
###############
st.subheader("Sobre as medições realizadas")
st.write("""A tomada das medições de temperatura e umidade foi feita utilizando um computador do tipo raspberry pi 3 com
aplicação do sensor DHT22 (datasheet disponível em https://pdf1.alldatasheet.com/datasheet-pdf/view/1132459/ETC2/DHT22.html).""")
st.write("""A rotina de captura está disponível no repositório github https://github.com/ciribelli/autohome/blob/master/motorHome.py.""")
st.write("""O período de amostragem dos dados é de sessenta (60) segundos, tanto para temperatura quanto para umidade, e as informações são registradas na base de dados sqlite3 no arquivo db.sqlite3.""")
image = Image.open('img_raspberry.jpg')
st.image(image, caption='Captura de medidas de temperatura e umidade utilizando raspberry pi e sensor DHT22', width=700)

st.write("""-----""")
#############
st.subheader("Sobre o formato dos dados brutos")
st.write("""O formato dos dados brutos está apresentado abaixo por meio de uma captura de tela do software 'DB Browser for SQLite'.""")
st.markdown(
"""
Algumas informações relevantes do banco de dados:
- são 603.117 registros para dois ambientes diferentes
- a coluna 'local_id' indica a origem do local das medições (são dois ambientes diferentes)
- a coluna data indica o timestamp do instante em que os registros são capturados do sensor
- temperaturas são registradas em graus celsius
- umidades são registradas em valores percentuais
"""
)
image = Image.open('img_db.png')
st.image(image, caption='Aspecto geral dos registros no visualizador SQLight', width=600)

st.write("""-----""")
###############
st.subheader("Sobre a composição do dataset para o trabalho proposto")
st.write("""Para a composição do dataset deste trabalho são tomadas medidas indiretas a partir dos dados brutos
apresentados. Essas medidas consistem do resultado de operações de janelas deslizantes de diferentes tamanhos 
aplicadas sobre os dados disponíveis.""")
st.write("""As operações são realizadas utilizando a função _rolling_ da biblioteca Pandas 
(https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html) com janelas que têm a seguinte composição:""")

code = '''
janela_0 = 15 # janela deslizante de 15 elementos
janela_1 = 30 # janela deslizante de 30 elementos
janela_2 = 60 # janela deslizante de 60 elementos

'''
st.code(code, language='python')

st.write("""Código-exemplo de aplicação das janelas ao dado bruto:""")

code = '''

l_janelas = [janela_0, janela_1, janela_2]

for l_j in l_janelas:

    # para temperaturas
    s_arcon_t = Y_t.rolling(l_j).std()              # desvio padrao
    v_arcon_t = Y_t.rolling(l_j).var()              # variância
    m_arcon_t = Y_t.rolling(l_j).mean()             # média
    min_arcon_t = Y_t.rolling(l_j).min()            # calc. auxiliar de mínimo
    max_arcon_t = Y_t.rolling(l_j).max()            # calc. auxiliar de máximo
    a_arcon_t = max_arcon_t - min_arcon_t           # amplitude (max - min)

'''
st.code(code, language='python')



@st.cache(max_entries=10, ttl=3600)
def load_sql(ambiente):
    con = sqlite3.connect("/Users/ciribelli/Server/Data Mining/db.sqlite3")
    saida = pd.read_sql_query("SELECT * from controleambiente_ambiente", con)
    amb = saida[saida.local_id == ambiente]
    
    con.close()
    return amb

# seta qual ambiente e chama a rotina
# ambiente = st.sidebar.number_input('Escolha o ambiente',0,1,1)
# ambiente = 0
# saida = load_sql(ambiente)
# saida['data'] =  pd.to_datetime(saida['data'], format='%Y-%m-%d %H:%M:%S')
# ambiente 0
# saida = saida[saida['data'].dt.date <= datetime.date(year=2018,month=12,day=6)]
# saida = saida[saida['data'].dt.date >= datetime.date(year=2018,month=12,day=5)]
# ambiente 0
# saida = saida[saida['data'].dt.date <= datetime.date(year=2018,month=10,day=4)]
# saida = saida[saida['data'].dt.date >= datetime.date(year=2018,month=10,day=3)]
#ambiente 0
# saida = saida[saida['data'].dt.date <= datetime.date(year=2018,month=11,day=29)]
# saida = saida[saida['data'].dt.date >= datetime.date(year=2018,month=11,day=28)]
#ambiente 1
# saida = saida[saida['data'].dt.date <= datetime.date(year=2020,month=4,day=8)]
# saida = saida[saida['data'].dt.date >= datetime.date(year=2020,month=4,day=7)]
#ambiente 1
#saida = saida[saida['data'].dt.date <= datetime.date(year=2020,month=4,day=12)]
#saida = saida[saida['data'].dt.date >= datetime.date(year=2020,month=4,day=11)]
#ambiente 1
# saida = saida[saida['data'].dt.date <= datetime.date(year=2020,month=4,day=13)]
# saida = saida[saida['data'].dt.date >= datetime.date(year=2020,month=4,day=12)]




################################################
## ---- novo bloco de aplicacao da janela ---- #
## ---- Load do SQL para ambiente 0       -----#
ambiente = 0
saida = load_sql(ambiente)
saida['data'] =  pd.to_datetime(saida['data'], format='%Y-%m-%d %H:%M:%S')
## ---                                    ____#

saida1 = saida[saida['data'].dt.date <= datetime.date(year=2018,month=12,day=6)]
saida1 = saida[saida['data'].dt.date >= datetime.date(year=2018,month=12,day=5)]
df = filtra(saida1)
df = aplica_janela(df)
'Dataset proposto no trabalho: ',
df = df[(df['tempo']>'2018-12-05 16:28') & (df['tempo']<'2018-12-06 14:00')]
df
'O número de colunas do dataset resultante é de: ', len(df.columns)
st.write("""-----""")
###############
st.subheader("Sobre a rotulagem dos dados")
st.write("""A sequência de gráficos apresentados abaixo mostra o aspecto das curvas de temperatura e umidade, bem como
a sinalização do estado do ar condicionado, se ligado ou desligado. Essas rotulagens foram feitas manualmente com base na
experiência e observação dos sinais frente aos comandos de liga e desliga dos aparelhos.""")


# AQUI ESTA COMENTADO PORQUE PRECISEI CHAMAR NA SESSAO ANTERIOR PARA DESCREVER O DATASET
# ambiente = 0
# saida = load_sql(ambiente)
# saida['data'] =  pd.to_datetime(saida['data'], format='%Y-%m-%d %H:%M:%S')
# saida = saida[saida['data'].dt.date <= datetime.date(year=2018,month=12,day=6)]
# saida = saida[saida['data'].dt.date >= datetime.date(year=2018,month=12,day=5)]
# df = aplica_janela(saida)
#------------------ INTERVALO 1 SELECIONADO ------------------------------
#                                                                        #
#                                                                        #       
# classificao supervisionada                                             #
#05/12/2018 16:28 ligou o ar
#06/12/2018 01:36 desligou
#06/12/2017 14:00 fim da janela
df = df[(df['tempo']>'2018-12-05 16:28') & (df['tempo']<'2018-12-06 14:00')]
one = np.ones(len(df))
df['classificacao'] = one
df['classificacao'] = np.where(df['tempo']>='2018-12-06 01:36', 0, df['classificacao'])
plota(df)

DF_Final = df

saida2 = saida[saida['data'].dt.date <= datetime.date(year=2018,month=10,day=4)]
saida2 = saida[saida['data'].dt.date >= datetime.date(year=2018,month=10,day=3)]
df = filtra(saida2)
df = aplica_janela(df)

#------------------ INTERVALO 2 SELECIONADO ------------------------------
#                                                                        #
#                                                                        #       
# classificao supervisionada                                             #
#03/10/2018 21:43 ligou o ar
#04/10/2018 02:43 desligou
#04/10/2017 22:40 fim da janela
df = df[(df['tempo']>'2018-10-03 21:43') & (df['tempo']<'2018-10-04 22:40')]
one = np.ones(len(df))
df['classificacao'] = one
df['classificacao'] = np.where(df['tempo']>='2018-10-04 02:43', 0, df['classificacao'])
plota(df)

DF_Final = pd.concat([DF_Final, df])




saida3 = saida[saida['data'].dt.date <= datetime.date(year=2018,month=11,day=29)]
saida3 = saida[saida['data'].dt.date >= datetime.date(year=2018,month=11,day=28)]
df = filtra(saida3)
df = aplica_janela(df)
#------------------ INTERVALO 3 SELECIONADO ------------------------------
#                                                                        #
#                                                                        #       
# classificao supervisionada                                             #
df = df[(df['tempo']>'2018-11-28 16:23') & (df['tempo']<'2018-11-29 22:40')]
one = np.ones(len(df))
df['classificacao'] = one
df['classificacao'] = np.where(df['tempo']>='2018-11-29 01:07', 0, df['classificacao'])
plota(df)

DF_Final = pd.concat([DF_Final, df])



## ---- Load do SQL para ambiente 0       -----#
ambiente = 1
saida = load_sql(ambiente)
saida['data'] =  pd.to_datetime(saida['data'], format='%Y-%m-%d %H:%M:%S')


saida4 = saida[saida['data'].dt.date <= datetime.date(year=2020,month=4,day=8)]
saida4 = saida[saida['data'].dt.date >= datetime.date(year=2020,month=4,day=7)]
df = filtra(saida4)
df = aplica_janela(saida4)
#------------------ INTERVALO 4 SELECIONADO ------------------------------
#07/04/2020 23:19 ligou o ar
#08/04/2020 10:30:51 desligou                                                   #                                                                     #       
# classificao supervisionada                                             #
df = df[(df['tempo']>'2020-04-07 23:19') & (df['tempo']<'2020-04-08 17:48')]
one = np.ones(len(df))
df['classificacao'] = one
df['classificacao'] = np.where(df['tempo']>='2020-04-08 10:30', 0, df['classificacao'])
plota(df)
DF_Final = pd.concat([DF_Final, df])



saida5 = saida[saida['data'].dt.date <= datetime.date(year=2020,month=4,day=8)]
saida5 = saida[saida['data'].dt.date >= datetime.date(year=2020,month=4,day=7)]
df = filtra(saida5)
df = aplica_janela(saida5)
#------------------ INTERVALO 5 SELECIONADO ------------------------------
#12/04/2020 23:30 ligou o ar
#13/04/2020 10:46:51 desligou
#13/04/2020 22:00:51 fim intervalo                       #                                                                     #       
# classificao supervisionada                                             #
df = df[(df['tempo']>'2020-04-12 23:30') & (df['tempo']<'2020-04-13 22:00')]
one = np.ones(len(df))
df['classificacao'] = one
df['classificacao'] = np.where(df['tempo']>='2020-04-13 10:46', 0, df['classificacao'])
plota(df)
DF_Final = pd.concat([DF_Final, df])


saida6 = saida[saida['data'].dt.date <= datetime.date(year=2020,month=4,day=8)]
saida6 = saida[saida['data'].dt.date >= datetime.date(year=2020,month=4,day=7)]
df = filtra(saida6)
df = aplica_janela(saida6)
#------------------ INTERVALO 6 SELECIONADO ------------------------------
#11/04/2020 01:23 ligou o ar
#11/04/2020 11:32 desligou
#11/04/2020 16:52 fim intervalo                      #                                                                     #       
# classificao supervisionada                                             #
df = df[(df['tempo']>'2020-04-11 01:23') & (df['tempo']<'2020-04-11 16:52')]
one = np.ones(len(df))
df['classificacao'] = one
df['classificacao'] = np.where(df['tempo']>='2020-04-11 11:32', 0, df['classificacao'])
plota(df)
DF_Final = pd.concat([DF_Final, df])


st.write("""-----""")
###############
st.subheader("Sobre as implementações propostas no Trabalho da Disciplina")

st.write("**- Análise exploratória e descarte de atributos desnecessários**")
st.write("""Esta etapa já foi iniciada por meio da organização dos dados, rotulagem e seleção de períodos 
de interesse. Também foram realizados testes rápidos com features não otimizadas para comprovar a hipótese de 
classificação por meio de _machine learning_. A proposição de diferentes tempos de janelamento (_rolling_) para as duas 
variáveis medidas também fez parte da análise exploratória dos dados.""")
st.write("""Essa etapa contará ainda com uma interpretação exploratória para seleção/descarte de atributos utilizando software RapidMiner e 
bibliotecas python indicadas na discplina de DM (Seaborn et al).""")


st.write("**- _Missing values_**")
st.write("""O trabalho deverá compreender análise de termos faltantes e filtragem de dados. As etapas de aquisição, 
por vezes, envolvem perda do dado ou medição errática que deverão ser tratadas nessa etapa com uma abordagem de pré-processamento.
Existem também casos de outliers e sensores que entraram e falha o degradação com o passar do tempo. No gráfico abaixo, é apresentado 
um caso de filtragem que será aplicada no trabalho. """)






#### velha FILTRAGEM #####
saida = saida[saida['data'].dt.date <= datetime.date(year=2018,month=12,day=31)]
saida = saida[saida['data'].dt.date >= datetime.date(year=2018,month=12,day=1)]
Y_t = saida['temperatura']
Y_u = saida['umidade']
T = saida['data']
df = pd.DataFrame({'tempo': T, 'temperatura': Y_t, 'umidade': Y_u})                                            #
df = df[(df['tempo']>'2018-1-1 00:00') & (df['tempo']<'2018-12-31 23:59')]

k = saida['temperatura']
removed = k.between(k.quantile(.002), k.quantile(.998))
#removed = k.between(k.quantile(.1), k.quantile(.9))
##'removed', removed

# saldo remanescente
##a = str(k[removed].size) + "/" + str(k.size) + " data points remain."
##a

#'# saldo',removed.value_counts()

# limpando o ventor apos inverter os removed outliers
filtrado = saida[~removed].index # INVERT removed_outliers!!
##'data para excluir:', filtrado
# 'saida', saida

nova = saida.drop(filtrado)
##'teste pandas drop:     ', nova
##############  ATENCAO PARA ESSE OFFSET DE 5.0 APENAS PARA INVESTIGACAO DE BUGGG
Y = nova['temperatura']
TEMPO = nova['data']

fig = go.Figure()
#fig.add_trace(go.Scatter(x=saida['data'], y=Y, mode='lines', name='dado bruto'))
fig.add_trace(go.Scatter(x=saida['data'], y=saida['temperatura'], mode='lines', name='sinal original'))
fig.add_trace(go.Scatter(x=TEMPO, y=Y, mode='lines', name='sinal filtrado', line={'color': 'green'}))
#fig.add_trace(go.Scatter(x=saida['data'], y=saida['umidade'], mode='lines', name='umidade'))
st.plotly_chart(fig)


st.write("**- Balanceamento e normalização**")
st.write("""Será realizada uma análise de balanceamento dos dados considerando as métricas de performance apresentadas
na disciplina de DM (precisão, revocação, F1 score, matriz de confusão, et al). Também serão considerados testes
de normalização para avaliar o impacto nos resultados do processo.""")

st.write("**- Testes de Modelos ML e transformação dos dados**")
st.write("""Por fim, será realizado uma sessão de testes compreensivos nos modelos de ML apresentados na
disciplina de ML. Para tal, serão levadas em conta eventuais necessidades de transformações nos dados. Dentre os métodos
de **classificação** a serem testados, pode-se citar SVM (e suas variações), _nearest neighbors_, regressão logística, árvores de decisão e Emsembles (Gradient Boosting, por exemplo).""")




st.subheader("Anexo")
st.write("**- Describe dos dados iniciais**")
'describe do df proposto', DF_Final.describe()

DF_Final.to_csv('/Users/ciribelli/Server/Data Mining/datasetDM.csv', index=False)







# ------  CLASSIFICANDO EM PREPARACAO AO SVM ----------------
janela = 30#12
janela2x = 60#20
s_arcon = Y.rolling(janela).std() # desvio padrao
v_arcon = Y.rolling(janela).var() # variancia
min_arcon = Y.rolling(janela2x).min()
max_arcon = Y.rolling(janela2x).max()
a_arcon = max_arcon - min_arcon # amplitude
m_arcon = a_arcon.rolling(janela2x).mean()
r_arcon = Y.rolling(60).mean() # real media


ndf = pd.DataFrame({'tempo': TEMPO, 'desvio padrao': s_arcon, 'variancia': v_arcon, 'amplitude': a_arcon, 'media real': r_arcon, 'temperatura': Y})
#'novodataframe ', ndf

############ TRUNCADO OS DADOS DO DATAFRAME PELA DATA ##################
#trunked new dataframe
#ML trainning
#07/04/2020 20:00 inicio (<<--- cancelando por ora)
#07/04/2020 23:19 ligou o ar
#08/04/2020 17:00 fim do trend 
#tndf = ndf[(ndf['tempo']>'2020-04-07 20:00') & (ndf['tempo']<'2020-04-08 17:00')] Essa tentativa falhou.. era tipo 000011110000.. 
#tndf = ndf[(ndf['tempo']>'2020-04-07 23:19') & (ndf['tempo']<'2020-04-08 17:00')]
#'trunkednovodataframe ', tndf

#------------------ INTERVALO 1 SELECIONADO ------------------------------
#                                                                        #
#                                                                        #       
# classificao supervisionada                                             #
#07/04/2020 23:19 ligou o ar
#08/04/2020 10:30:51 desligou


#------------------ INTERVALO 2 SELECIONADO ------------------------------
#                                                                        #
#                                                                        #       
# classificao supervisionada                                             #
#12/04/2020 23:30 ligou o ar
#13/04/2020 10:46:51 desligou
#13/04/2020 22:00:51 fim intervalo

#------------------ INTERVALO 3 SELECIONADO ------------------------------
#                                                                        #
#                                                                        #       
# classificao supervisionada                                             #
#11/04/2020 01:23 ligou o ar
#11/04/2020 11:32 desligou
#11/04/2020 16:52 fim intervalo

tndf = ndf[(ndf['tempo']>'2020-04-11 01:23') & (ndf['tempo']<'2020-04-11 16:52')]
one = np.ones(len(tndf))
tndf['classificacao'] = one
#'novodataframe com ones', tndf

tndf['classificacao'] = np.where(tndf['tempo']>='2020-04-11 11:32:00', 0, tndf['classificacao'])

#novodataframe com classificacao', tndf    
# SALVANDO
tndf.to_csv('/Users/ciribelli/Server/files/DM_saida_04.csv', index=False)
print("arquivo salvo")

# Create figure with secondary y-axis
# fig2 = make_subplots(specs=[[{"secondary_y": True}]])
# fig2.add_trace(go.Scatter(x=TEMPO, y = s_arcon, mode='lines', name='std'),secondary_y=False)
# fig2.add_trace(go.Scatter(x=TEMPO, y=v_arcon, mode='lines', name='var'),secondary_y=False)
# fig2.add_trace(go.Scatter(x=TEMPO, y=a_arcon, mode='lines', name='amplitude'),secondary_y=False)
# fig2.add_trace(go.Scatter(x=TEMPO, y=m_arcon, mode='lines', name='media'),secondary_y=False)
# fig2.add_trace(go.Scatter(x=TEMPO, y=r_arcon, mode='lines', name='real mean'),secondary_y=True)
# fig2.add_trace(go.Scatter(x=TEMPO, y=Y, mode='lines', name='filtrado'),secondary_y=True)
# st.plotly_chart(fig2)


