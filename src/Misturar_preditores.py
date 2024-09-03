'''
    colunas_func_cog = [
    'Último item com tentativa completa',
    'Total score',
    'Total tentativas correctas',
    'Memory span',
    'Total corretos',
    'Total erros',
    'Acuracia Media',
    'Erros media',
    'Acuracia Go',
    'Acuracia nogo',
    'Tempo resposta Go',
    'Tempo resposta nogo',
    'Media de acertos',
    'Capacidade de rastreamento',
    'Melhor tempo total A',
    'Melhor tempo total B',
    'Flexibilidade cognitiva (B-A)',
    'Acuracia media A',
    'Acuracia media B',
    'Media tempo total A',
    'Media tempo total B',
    'Flexibilidade cognitiva media (B-A)'
    ]
'''

#Importando as bibliotecas necessÃ¡rias
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle
from imblearn.over_sampling import SMOTE
from scipy import stats
import os
from ydata_profiling import ProfileReport
from scipy.stats import linregress
from itertools import combinations
 
# Função para gerar todas as combinações possíveis
def gerar_combinacoes(colunas):
    todas_combinacoes = []
    for r in range(1, len(colunas) + 1):
        comb = list(combinations(colunas, r))
        todas_combinacoes.extend(comb)
    return todas_combinacoes       

def kmeans_cluster(dados, n_clusters):

    # Inicializa o modelo K-means com o número de clusters desejado
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    
    # Aplica o modelo aos dados
    kmeans.fit(dados)
    
    # Adiciona os rótulos dos clusters aos dados originais
    dados_com_clusters = dados.copy()
    dados_com_clusters['cluster'] = kmeans.labels_
    
    return dados_com_clusters

def kmeans_cluster_teams(dados, n_clusters):
    
    #seleciona dados time 1
    dados_1team = dados[:22].copy()
    
    # Normaliza dados time 1 e transforma em dataframe
    scaler_1team = StandardScaler()
    dados_1team_norm = scaler_1team.fit_transform(dados_1team)
    dados_1team_norm_pd = pd.DataFrame(dados_1team_norm)
    
    # Inicializa o modelo K-means com o número de clusters desejado
    dados_team_1 = kmeans_cluster(dados_1team_norm_pd, n_clusters)
    dados_team_1 = reorganizar_labels(dados_team_1, n_clusters)
    
    #seleciona dados time 2
    dados_2team = dados[22:].copy()
    
    # Normaliza dados time 1 e transforma em dataframe
    scaler_2team = StandardScaler()
    dados_2team_norm = scaler_2team.fit_transform(dados_2team)
    dados_2team_norm_pd = pd.DataFrame(dados_2team_norm)
    
    # Inicializa o modelo K-means com o número de clusters desejado
    dados_team_2 = kmeans_cluster(dados_2team_norm_pd, n_clusters)
    dados_team_2 = reorganizar_labels(dados_team_2, n_clusters)
    
    # Concatena os DataFrames resultantes
    dados_com_clusters = pd.concat([dados_team_1, dados_team_2])
    
    return dados_com_clusters

def add_cluster_metade(dados):
    # Ordenar os dados em ordem decrescente
    d_sorted = dados.sort_values(by=0, ascending=False)
    
    # Determinar o ponto de corte para metade dos dados
    midpoint = len(d_sorted) // 2
    
    # Adicionar a coluna 'cluster'
    d_sorted['cluster'] = 0
    d_sorted.iloc[:midpoint, d_sorted.columns.get_loc('cluster')] = 1
    
    # Voltar a ordenar pelo índice original
    d_sorted = d_sorted.sort_index()
    
    return d_sorted

def label_metade_teams(dados, n_clusters):
    
    #seleciona dados time 1
    dados_1team = dados[:22].copy()
    
    # Normaliza dados time 1 e transforma em dataframe
    scaler_1team = StandardScaler()
    dados_1team_norm = scaler_1team.fit_transform(dados_1team)
    dados_1team_norm_pd = pd.DataFrame(dados_1team_norm)
    
    # Inicializa o modelo K-means com o número de clusters desejado
    dados_team_1 = add_cluster_metade(dados_1team_norm_pd)
    
    #seleciona dados time 2
    dados_2team = dados[22:].copy()
    
    # Normaliza dados time 2 e transforma em dataframe
    scaler_2team = StandardScaler()
    dados_2team_norm = scaler_2team.fit_transform(dados_2team)
    dados_2team_norm_pd = pd.DataFrame(dados_2team_norm)
    
    # Inicializa o modelo K-means com o número de clusters desejado
    dados_team_2 = add_cluster_metade(dados_2team_norm_pd)
    
    # Concatena os DataFrames resultantes
    dados_com_clusters = pd.concat([dados_team_1, dados_team_2])
    
    return dados_com_clusters

def kmeans_cluster_columns(dados, n_clusters):
    dados_com_clusters = pd.DataFrame(index=dados.index)  # DataFrame para armazenar os clusters
    
    # Iterar sobre todas as colunas
    for coluna in dados.columns:
        # Selecionar apenas a coluna atual para aplicar o KMeans
        coluna_dados = dados[[coluna]]
        
        # Inicializar e ajustar o modelo KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(coluna_dados)
        
        # Adicionar rótulos de cluster ao DataFrame de saída
        dados_com_clusters[str(coluna) + '_cluster'] = kmeans.labels_
    
    return dados_com_clusters

def reorganizar_labels(dados, n_clusters):
    
    # Ordena os dados pela primeira coluna em ordem decrescente
    dados_ordenados = dados.sort_values(by=dados.columns[0], ascending=False)
    
    # Identifica a ordem original dos rótulos dos clusters
    ordem_original = dados_ordenados[dados_ordenados.columns[1]].unique()
    
    # Define a nova ordem dos rótulos dos clusters como 2, 1, 0
    nova_ordem = list(range(n_clusters - 1, -1, -1))
    
    dados_reorganizados = dados_ordenados.copy()
    # Substitui os rótulos dos clusters pela nova ordem
    for i, label in enumerate(ordem_original):
        dados_reorganizados[dados_reorganizados.columns[1]][dados_ordenados[dados_ordenados.columns[1]] == label] = nova_ordem[i]
    
    # Reorganiza o DataFrame de acordo com o índice original
    dados_reorganizados = dados_reorganizados.sort_index()
    
    # Retorna os dados reorganizados
    return dados_reorganizados


def run(path_to_data, path_save, n_clusters_Desempenho_campo = 2, oversample = False, normal = False, metade = False, colunas_func_cog = None):
    
    full_data = pd.read_csv(path_to_data)
    
    #variavel preditora
    ###########para utilizar todos dados#############
    #X_cognitivo = full_data.iloc[:, 12:]
    
    ##########selecionando somente funções cognitivas#############
    X_cognitivo = full_data[colunas_func_cog]
    
    #normalizando dados
    scaler_cog = StandardScaler()
    X_cognitivo_norm = scaler_cog.fit_transform(X_cognitivo)
    
    #Selecionando colunas com dados de desempenho em campo
    Y_campo = full_data.iloc[:, :12]
    colunas_campo = ['gols_feitos', 'gols_sofridos', 'gols_companheiros', 'saldo_gols']

    #Transformando em pandas para rodar o elbow method
    X_cognitivo_pd = pd.DataFrame(X_cognitivo_norm)
    
    #Investigar em quantos clusters é correto dividir os dados utilizando o método do cotovelo (Elbow Method)
    colunas_cog = X_cognitivo_pd.columns
    
    #Normalizando os dados e aplicar cluster para os dados do desempenho em campo
    Y_gols_feitos = kmeans_cluster_teams(Y_campo[['gols_feitos']], n_clusters_Desempenho_campo)
    Y_gols_sofridos = kmeans_cluster_teams(Y_campo[['gols_sofridos']], n_clusters_Desempenho_campo)
    Y_gols_companheiros = kmeans_cluster_teams(Y_campo[['gols_companheiros']], n_clusters_Desempenho_campo)
    Y_saldo_gols = kmeans_cluster_teams(Y_campo[['saldo_gols']], n_clusters_Desempenho_campo)
    
    # Crie o diretório para salvar se ele não existir
    os.makedirs(path_save, exist_ok=True)
    
    #################Aplicando sobreamostragem (oversample) nos dados de campo para balanceamento###############
    #Para oversample salvar dados sem normalização, para que no treino possa ser feito o oversample e a normalização somente nos dados de treino
    if oversample == True:
        
        #Separando dados em teste e treino gols feitos
        X_cognitivo_treinamento, X_cognitivo_teste, Y_gols_feitos_treinamento, Y_gols_feitos_teste = train_test_split(X_cognitivo, Y_gols_feitos, test_size = 0.25, random_state = 0)
   
        with open(os.path.join(path_save, 'cogfut_gf.pkl'), mode = 'wb') as f:
            pickle.dump([X_cognitivo_treinamento, Y_gols_feitos_treinamento['cluster'], X_cognitivo_teste, Y_gols_feitos_teste['cluster']], f)
            
        #Separando dados em teste e treino gols sofridos
        X_cognitivo_treinamento, X_cognitivo_teste, Y_gols_sofridos_treinamento, Y_gols_sofridos_teste = train_test_split(X_cognitivo, Y_gols_sofridos, test_size = 0.25, random_state = 0)
        
        with open(os.path.join(path_save, 'cogfut_gs.pkl'), mode = 'wb') as f:
            pickle.dump([X_cognitivo_treinamento, Y_gols_sofridos_treinamento['cluster'], X_cognitivo_teste, Y_gols_sofridos_teste['cluster']], f)
        
        #Separando dados em teste e treino gols companheiros
        X_cognitivo_treinamento, X_cognitivo_teste,  Y_gols_companheiros_treinamento, Y_gols_companheiros_teste = train_test_split(X_cognitivo,  Y_gols_companheiros, test_size = 0.25, random_state = 0)
        
        with open(os.path.join(path_save, 'cogfut_gc.pkl'), mode = 'wb') as f:
            pickle.dump([X_cognitivo_treinamento,  Y_gols_companheiros_treinamento['cluster'], X_cognitivo_teste,  Y_gols_companheiros_teste['cluster']], f)
        
        #Separando dados em teste e treino saldo de gols
        X_cognitivo_treinamento, X_cognitivo_teste, Y_saldo_gols_treinamento, Y_saldo_gols_teste = train_test_split(X_cognitivo, Y_saldo_gols, test_size = 0.25, random_state = 0)
        
        with open(os.path.join(path_save, 'cogfut_sg.pkl'), mode = 'wb') as f:
            pickle.dump([X_cognitivo_treinamento, Y_saldo_gols_treinamento['cluster'], X_cognitivo_teste, Y_saldo_gols_teste['cluster']], f)
        
        return()
    
    ###############################################################################################################
        
    if normal == True:
        #Separando dados em teste e treino gols feitos
        X_cognitivo_treinamento, X_cognitivo_teste, Y_gols_feitos_treinamento, Y_gols_feitos_teste = train_test_split(X_cognitivo_norm, Y_gols_feitos, test_size = 0.25, random_state = 0)
   
        with open(os.path.join(path_save, 'cogfut_gf.pkl'), mode = 'wb') as f:
            pickle.dump([X_cognitivo_treinamento, Y_gols_feitos_treinamento['cluster'], X_cognitivo_teste, Y_gols_feitos_teste['cluster']], f)
            
        #Separando dados em teste e treino gols sofridos
        X_cognitivo_treinamento, X_cognitivo_teste, Y_gols_sofridos_treinamento, Y_gols_sofridos_teste = train_test_split(X_cognitivo_norm, Y_gols_sofridos, test_size = 0.25, random_state = 0)
        
        with open(os.path.join(path_save, 'cogfut_gs.pkl'), mode = 'wb') as f:
            pickle.dump([X_cognitivo_treinamento, Y_gols_sofridos_treinamento['cluster'], X_cognitivo_teste, Y_gols_sofridos_teste['cluster']], f)
    
        #Separando dados em teste e treino gols companheiros
        X_cognitivo_treinamento, X_cognitivo_teste,  Y_gols_companheiros_treinamento, Y_gols_companheiros_teste = train_test_split(X_cognitivo_norm,  Y_gols_companheiros, test_size = 0.25, random_state = 0)
        
        with open(os.path.join(path_save, 'cogfut_gc.pkl'), mode = 'wb') as f:
            pickle.dump([X_cognitivo_treinamento,  Y_gols_companheiros_treinamento['cluster'], X_cognitivo_teste,  Y_gols_companheiros_teste['cluster']], f)
    
        #Separando dados em teste e treino saldo de gols
        X_cognitivo_treinamento, X_cognitivo_teste, Y_saldo_gols_treinamento, Y_saldo_gols_teste = train_test_split(X_cognitivo_norm, Y_saldo_gols, test_size = 0.25, random_state = 0)
        
        with open(os.path.join(path_save, 'cogfut_sg.pkl'), mode = 'wb') as f:
            pickle.dump([X_cognitivo_treinamento, Y_saldo_gols_treinamento['cluster'], X_cognitivo_teste, Y_saldo_gols_teste['cluster']], f)
            
    if metade == True:
        
        #Normalizando os dados e aplicando cluster na metade para os dados do desempenho em campo
        Y_gols_feitos = label_metade_teams(Y_campo[['gols_feitos']], n_clusters_Desempenho_campo)
        Y_gols_sofridos = label_metade_teams(Y_campo[['gols_sofridos']], n_clusters_Desempenho_campo)
        Y_gols_companheiros = label_metade_teams(Y_campo[['gols_companheiros']], n_clusters_Desempenho_campo)
        Y_saldo_gols = label_metade_teams(Y_campo[['saldo_gols']], n_clusters_Desempenho_campo)
        
        #Separando dados em teste e treino gols feitos
        X_cognitivo_treinamento, X_cognitivo_teste, Y_gols_feitos_treinamento, Y_gols_feitos_teste = train_test_split(X_cognitivo_norm, Y_gols_feitos, test_size = 0.25, random_state = 0)
   
        with open(os.path.join(path_save, 'cogfut_gf.pkl'), mode = 'wb') as f:
            pickle.dump([X_cognitivo_treinamento, Y_gols_feitos_treinamento['cluster'], X_cognitivo_teste, Y_gols_feitos_teste['cluster']], f)
            
        #Separando dados em teste e treino gols sofridos
        X_cognitivo_treinamento, X_cognitivo_teste, Y_gols_sofridos_treinamento, Y_gols_sofridos_teste = train_test_split(X_cognitivo_norm, Y_gols_sofridos, test_size = 0.25, random_state = 0)
        
        with open(os.path.join(path_save, 'cogfut_gs.pkl'), mode = 'wb') as f:
            pickle.dump([X_cognitivo_treinamento, Y_gols_sofridos_treinamento['cluster'], X_cognitivo_teste, Y_gols_sofridos_teste['cluster']], f)
    
        #Separando dados em teste e treino gols companheiros
        X_cognitivo_treinamento, X_cognitivo_teste,  Y_gols_companheiros_treinamento, Y_gols_companheiros_teste = train_test_split(X_cognitivo_norm,  Y_gols_companheiros, test_size = 0.25, random_state = 0)
        
        with open(os.path.join(path_save, 'cogfut_gc.pkl'), mode = 'wb') as f:
            pickle.dump([X_cognitivo_treinamento,  Y_gols_companheiros_treinamento['cluster'], X_cognitivo_teste,  Y_gols_companheiros_teste['cluster']], f)
    
        #Separando dados em teste e treino saldo de gols
        X_cognitivo_treinamento, X_cognitivo_teste, Y_saldo_gols_treinamento, Y_saldo_gols_teste = train_test_split(X_cognitivo_norm, Y_saldo_gols, test_size = 0.25, random_state = 0)
        
        with open(os.path.join(path_save, 'cogfut_sg.pkl'), mode = 'wb') as f:
            pickle.dump([X_cognitivo_treinamento, Y_saldo_gols_treinamento['cluster'], X_cognitivo_teste, Y_saldo_gols_teste['cluster']], f)

if __name__ == "__main__":
    
    colunas_func_cog = ['Memory span', 'Acuracia Go', 'Acuracia nogo', 'Capacidade de rastreamento', 'Flexibilidade cognitiva (B-A)']
    
    # Gerando as combinações
    combinacoes = gerar_combinacoes(colunas_func_cog)
    
    path_base_save = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\data\\full_cog_misturado_oversample'
    
    # Exibindo as combinações
    for combinacao in combinacoes:
        # Criando o nome do diretório com base na combinação
        combinacao_str = '_'.join([col.replace(' ', '_') for col in combinacao])
        
        # Caminho completo de salvamento
        path_save = os.path.join(path_base_save, combinacao_str)
        
        run(path_to_data = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\dataset.csv', 
            path_save = path_save,
            n_clusters_Desempenho_campo = 2, oversample = True, normal = False, metade = False, 
            colunas_func_cog = list(combinacao))
    