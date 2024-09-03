# Importando as bibliotecas necessÃ¡rias
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle
from imblearn.over_sampling import SMOTE


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

def teste_t(dados, tentativa):
    
    
    return()

def medias(data, tentativa):
    print('\n##########  Media  #############', tentativa)
    print(data.mean())
    
    print('\nMaior média:', data.mean().max(), 'na coluna', data.mean().idxmax())
    
    return()


def independent_t_test_summary(df1, df2, nome):
    
    print('\n##########  Teste t amostras independentes  #############', nome)
    
    results = {
        'Variable': [],
        'Média Melhores': [],
        'Média Piores': [],
        'Std Dev1': [],
        'Std Dev2': [],
        'T-statistic': [],
        'P-value': [],
        "Cohen's d": []
    }

    for col in df1.columns:
        data1 = df1[col].dropna()
        data2 = df2[col].dropna()

        t_statistic, p_value = stats.ttest_ind(data1, data2)
        
        mean1 = np.mean(data1)
        mean2 = np.mean(data2)
        std_dev1 = np.std(data1, ddof=1)
        std_dev2 = np.std(data2, ddof=1)
        
        pooled_std_dev = np.sqrt(((std_dev1 ** 2 + std_dev2 ** 2) / 2))
        cohen_d = (mean1 - mean2) / pooled_std_dev
        
        results['Variable'].append(col)
        results['Média Melhores'].append(mean1)
        results['Média Piores'].append(mean2)
        results['Std Dev1'].append(std_dev1)
        results['Std Dev2'].append(std_dev2)
        results['T-statistic'].append(t_statistic)
        results['P-value'].append(p_value)
        results["Cohen's d"].append(cohen_d)
    
    df_results = pd.DataFrame(results)
    
    # Configurar pandas para exibir todas as colunas
    pd.set_option('display.max_columns', None)  # Mostra todas as colunas
    pd.set_option('display.width', 1000)        # Define a largura para uma quantidade maior de colunas
    pd.set_option('display.max_colwidth', None) # Não limita a largura das colunas individuais

    print(df_results)
    
def get_top_n_indices(df, column, n, largest=True):
    """
    Ordena o DataFrame com base nos valores de uma coluna e retorna os índices dos n maiores ou menores valores.

    Parâmetros:
    df (pd.DataFrame): O DataFrame de entrada.
    column (str): O nome da coluna para ordenar.
    n (int): O número de índices para retornar.
    largest (bool): Se True, retorna os maiores valores. Se False, retorna os menores valores.

    Retorna:
    pd.Index: Índices dos n maiores ou menores valores.
    """
    if largest:
        sorted_df = df.sort_values(by=column, ascending=False)
        top_n_indices = sorted_df.head(n).index
    else:
        sorted_df = df.sort_values(by=column, ascending=True)
        bottom_n_indices = sorted_df.head(n).index
    
    return top_n_indices if largest else bottom_n_indices



def run(n_clusters_Desempenho_campo = 2, n_clusters_cognitivo = 7, cluster = True, oversample = False):
    path_to_data = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\full_data.csv'
    
    full_data = pd.read_csv(path_to_data)
    
    full_data = full_data.drop(columns=['Numero', 'Unnamed: 0', 'Numero.1', 'Numero.2', 'Numero.3', 'Numero.4'])
    
    lista_para_corrigir = ['Último item com tentativa completa', 'Total score', 'Total tentativas correctas', 'Memory span']
    for correct in lista_para_corrigir:
        full_data.loc[full_data[correct] < 2, correct] = full_data[correct][full_data[correct] > 2].mean()
    
    
    #variavel preditora
    ###########para utilizar todos dados#############
    #X_cognitivo = full_data.iloc[:, 12:]
    
    ##########somente funções cognitivas#############
    colunas_func_cog = ['Memory span', 'Acuracia Go', 'Acuracia nogo', 'Capacidade de rastreamento', 'Flexibilidade cognitiva (B-A)']
    X_cognitivo = full_data[colunas_func_cog]
    
    #varaivel de classe 
    Y_campo = full_data.iloc[:, :12]
    
    
    #Normalizando os dados e aplicar cluster para os dados do desempenho em campo
    Y_gols_feitos = kmeans_cluster_teams(Y_campo[['gols_feitos']], n_clusters_Desempenho_campo)
    Y_gols_sofridos = kmeans_cluster_teams(Y_campo[['gols_sofridos']], n_clusters_Desempenho_campo)
    Y_gols_companheiros = kmeans_cluster_teams(Y_campo[['gols_companheiros']], n_clusters_Desempenho_campo)
    Y_saldo_gols = kmeans_cluster_teams(Y_campo[['saldo_gols']], n_clusters_Desempenho_campo)
    
    # Garantir que ambos os DataFrames estão alinhados
    X_cognitivo = X_cognitivo.reset_index(drop=True)
    Y_gols_feitos = Y_gols_feitos.reset_index(drop=True)
    Y_gols_sofridos = Y_gols_sofridos.reset_index(drop=True)
    Y_gols_companheiros = Y_gols_companheiros.reset_index(drop=True)
    Y_saldo_gols = Y_saldo_gols.reset_index(drop=True)
    
    #rodando teste t
    gf_melhor_cog = X_cognitivo[Y_gols_feitos['cluster']==1]
    gf_pior_cog = X_cognitivo[Y_gols_feitos['cluster']==0]
    
    gs_melhor_cog = X_cognitivo[Y_gols_sofridos['cluster']==0]
    gs_pior_cog = X_cognitivo[Y_gols_sofridos['cluster']==1]
    
    gc_melhor_cog = X_cognitivo[Y_gols_companheiros['cluster']==1]
    gc_pior_cog = X_cognitivo[Y_gols_companheiros['cluster']==0]
    
    sg_melhor_cog = X_cognitivo[Y_saldo_gols['cluster']==1]
    sg_pior_cog = X_cognitivo[Y_saldo_gols['cluster']==0]
    

    ################## Medias #######################
    medias(gf_melhor_cog, 'gf_melhor_cog')
    medias(gf_pior_cog, 'gf_pior_cog')
    medias(gs_melhor_cog, 'gs_melhor_cog')
    medias(gs_pior_cog, 'gs_pior_cog')
    medias(gc_melhor_cog, 'gc_melhor_cog')
    medias(gc_pior_cog, 'gc_pior_cogl')
    medias(sg_melhor_cog, 'sg_melhor_cog')
    medias(sg_pior_cog, 'sg_pior_cog')
    
    ########### teste t independente ###############
    independent_t_test_summary(gf_melhor_cog, gf_pior_cog, 'Gols feitos')
    independent_t_test_summary(gs_melhor_cog, gs_pior_cog, 'Gols sofridos')
    independent_t_test_summary(gc_melhor_cog, gc_pior_cog, 'Gols companheiros')
    independent_t_test_summary(sg_melhor_cog, sg_pior_cog, 'Saldo Gols')
    
    print('\n##########  Teste t amostras independentes grupos balanceados  #############')
    
    #################################### teste t independente melhores e piores balanceados ########################################
    gf_pior_cog_balanceado_index = get_top_n_indices(Y_campo, 'gols_feitos', len(gf_melhor_cog), largest= False)
    gf_pior_cog_balanceado = gf_pior_cog.loc[gf_pior_cog_balanceado_index]
    
    gs_pior_cog_balanceado_index = get_top_n_indices(Y_campo, 'gols_sofridos', len(gs_melhor_cog), largest= True)
    gs_pior_cog_balanceado = gs_pior_cog.loc[gs_pior_cog_balanceado_index]
    
    gc_melhor_cog_balanceado_index = get_top_n_indices(Y_campo, 'gols_companheiros', len(gc_pior_cog), largest= True)
    gc_melhor_cog_balanceado = gc_melhor_cog.loc[gc_melhor_cog_balanceado_index]
    
    sg_pior_cog_balanceado_index = get_top_n_indices(Y_campo, 'saldo_gols', len(sg_melhor_cog), largest= False)
    sg_pior_cog_balanceado = sg_pior_cog.loc[sg_pior_cog_balanceado_index]
    
    ########### teste t independente ###############
    independent_t_test_summary(gf_melhor_cog, gf_pior_cog_balanceado, 'Gols feitos')
    independent_t_test_summary(gs_melhor_cog, gs_pior_cog_balanceado, 'Gols sofridos')
    independent_t_test_summary(gc_melhor_cog_balanceado, gc_pior_cog, 'Gols companheiros')
    independent_t_test_summary(sg_melhor_cog, sg_pior_cog_balanceado, 'Saldo Gols')
    
if __name__ == "__main__":
    
    run(n_clusters_Desempenho_campo = 2, n_clusters_cognitivo = 7, cluster = False, oversample = True)
    
    
    
    

    
    
    
    