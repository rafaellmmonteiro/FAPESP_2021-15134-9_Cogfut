# Importando as bibliotecas necessÃ¡rias
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
from scipy.stats import spearmanr, ttest_ind

def cohen_d_from_spearman(rho):
    """Calcula Cohen's d a partir do coeficiente de Spearman rho."""
    d = 2 * rho / np.sqrt(1 - rho**2)
    return d

def correlacao_spearman(full_data, print=False, save_path=None):
    """
    Calcula a correlação de Spearman entre as colunas do DataFrame, 
    plota um heatmap, fornece os valores p e calcula o d de Cohen para cada par de variáveis.

    Parâmetros:
    full_data (pd.DataFrame): DataFrame contendo os dados.
    print (bool): Se True, o heatmap será exibido; se False, apenas será salvo.
    save_path (str): Caminho onde o gráfico será salvo.
    """
    # Calcular a correlação de Spearman
    correlation_matrix, p_values_matrix = spearmanr(full_data, axis=0)
    
    # Convertendo as matrizes para DataFrames
    correlation_df = pd.DataFrame(correlation_matrix, index=full_data.columns, columns=full_data.columns)
    p_values_df = pd.DataFrame(p_values_matrix, index=full_data.columns, columns=full_data.columns)
    
    if print:
        # Plotar o heatmap da correlação de Spearman
        plt.figure(figsize=(14, 14))
        plt.title('Spearman\'s correlation', size=15)
        sns.heatmap(correlation_df, annot=True, cmap="RdYlGn", vmin=-1, vmax=1, fmt='.2f',
                    annot_kws={"size": 10})
        plt.xticks(rotation=45, ha='right', size=10)
        plt.yticks(size=10)
        plt.tight_layout()
        plt.show()
        
        # Plotar o heatmap dos valores de p
        plt.figure(figsize=(14, 14))
        plt.title('Valores de p da Correlação de Spearman', size=15)
        sns.heatmap(p_values_df, annot=True, cmap="RdYlBu_r", vmin=0, vmax=0.05, fmt='.2g',
                    annot_kws={"size": 10})
        plt.xticks(rotation=45, ha='right', size=10)
        plt.yticks(size=10)
        plt.tight_layout()
        plt.show()
    else:
        # Verificar se o caminho de salvamento foi fornecido
        if save_path is None:
            raise ValueError("Você deve fornecer um caminho para salvar o gráfico.")
        
        # Criar a pasta, se não existir
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Salvar o heatmap da correlação de Spearman
        plt.figure(figsize=(14, 14))
        plt.title('Spearman\'s correlation', size=15)
        sns.heatmap(correlation_df, annot=True, cmap="RdYlGn", vmin=-1, vmax=1, fmt='.2f',
                    annot_kws={"size": 10})
        plt.xticks(rotation=45, ha='right', size=10)
        plt.yticks(size=10)
        plt.tight_layout()
        plt.savefig(f"{save_path}_correlation.png")
        plt.close()
        
        # Salvar o heatmap dos valores de p
        plt.figure(figsize=(14, 14))
        plt.title('Valores de p da Correlação de Spearman', size=15)
        sns.heatmap(p_values_df, annot=True, cmap="RdYlBu_r", vmin=0, vmax=0.05, fmt='.2g',
                    annot_kws={"size": 10})
        plt.xticks(rotation=45, ha='right', size=10)
        plt.yticks(size=10)
        plt.tight_layout()
        plt.savefig(f"{save_path}_p_values.png")
        plt.close()

    # Calcular o d de Cohen para cada par de variáveis
    cohen_d_results = {}
    columns = full_data.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1 = columns[i]
            col2 = columns[j]
            rho = correlation_df.loc[col1, col2]
            d = cohen_d_from_spearman(rho)
            cohen_d_results[(col1, col2)] = d
    
    # Salvar os resultados do d de Cohen em um arquivo de texto
    if save_path is not None:
        with open(f"{save_path}_cohen_d_results.txt", "w") as f:
            f.write("Par de Variáveis, Cohen's d\n")
            for pair, d_value in cohen_d_results.items():
                f.write(f"{pair}, {d_value:.6f}\n")
    
    return correlation_df, p_values_df, cohen_d_results

def correlacao_pearson(full_data, print=False, save_path=None):
    """
    Calcula a correlação de Pearson entre as colunas do DataFrame e plota um heatmap.

    Parâmetros:
    full_data (pd.DataFrame): DataFrame contendo os dados.
    print (bool): Se True, o heatmap será exibido; se False, apenas será salvo.
    save_path (str): Caminho onde o gráfico será salvo.
    """
    # Calcular a correlação de Pearson
    correlation_matrix = full_data.corr(method='pearson')
    
    if print:
        # Plotar o heatmap
        plt.figure(figsize=(25, 15))
        plt.title('Correlação de Pearson', size=15)
        sns.heatmap(correlation_matrix, annot=True, cmap="RdYlGn")
        plt.show()
    else:
        # Verificar se o caminho de salvamento foi fornecido
        if save_path is None:
            raise ValueError("Você deve fornecer um caminho para salvar o gráfico.")
        
        # Criar a pasta, se não existir
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Salvar o heatmap
        plt.figure(figsize=(25, 15))
        plt.title('Correlação de Pearson', size=15)
        sns.heatmap(correlation_matrix, annot=True, cmap="RdYlGn")
        plt.savefig(save_path)
        plt.close()
        
def dispersao_grupos(data_x, data_y, clusters, path_save=None):
    
    clusters = clusters.reset_index(drop=True)
    
    path_save_folder = os.path.join(path_save, 'dispersao_grupos', f'{data_y.name}')
    os.makedirs(path_save_folder, exist_ok=True)
    
    # Plotando os dados do Grupo '0'
    for coluna_x in data_x.columns:
        
        grupo_0x = data_x.loc[clusters==0, coluna_x]
        grupo_1x = data_x.loc[clusters==1, coluna_x]
        
        #cruzando dados cognitivos
        for coluna_y in data_x.columns:
            
            grupo_0y = data_x.loc[clusters==0, coluna_y]
            grupo_1y = data_x.loc[clusters==1, coluna_y]
            
            # Criando o gráfico de dispersão
            plt.figure(figsize=(10, 6))
            
            plt.scatter(grupo_0x, grupo_0y, color='blue', label='Grupo 0')
            plt.scatter(grupo_1x, grupo_1y, color='red', label='Group 1')
            
            # Calculando a reta de regressão para o Grupo 0
            slope_0, intercept_0, _, _, _ = linregress(grupo_0x, grupo_0y)
            plt.plot(grupo_0x, slope_0 * grupo_0x + intercept_0, color='blue', linestyle='--')
            
            # Calculando a reta de regressão para o Grupo 1
            slope_1, intercept_1, _, _, _ = linregress(grupo_1x, grupo_1y)
            plt.plot(grupo_1x, slope_1 * grupo_1x + intercept_1, color='red', linestyle='--')
            
            # Adicionando rótulos e título
            plt.xlabel(coluna_x)
            plt.ylabel(coluna_y)
            plt.title(f'Dispersão {coluna_x} x {coluna_y}')
            plt.legend()
            
            # Salvando o gráfico
            figure_name = f'{coluna_x} x {coluna_y}.png'
            plt.savefig(os.path.join(path_save_folder, figure_name))
            plt.close()
        
        #plotando desempenho em campo
        grupo_0y = data_y.loc[clusters==0]
        grupo_1y = data_y.loc[clusters==1]
        
        # Criando o gráfico de dispersão
        plt.figure(figsize=(10, 6))
            
        plt.scatter(grupo_0x, grupo_0y, color='blue', label='Grupo 0')
        plt.scatter(grupo_1x, grupo_1y, color='red', label='Group 1')
        
        # Calculando a reta de regressão para o Grupo 0
        slope_0, intercept_0, _, _, _ = linregress(grupo_0x, grupo_0y)
        plt.plot(grupo_0x, slope_0 * grupo_0x + intercept_0, color='blue', linestyle='--')
            
        # Calculando a reta de regressão para o Grupo 1
        slope_1, intercept_1, _, _, _ = linregress(grupo_1x, grupo_1y)
        plt.plot(grupo_1x, slope_1 * grupo_1x + intercept_1, color='red', linestyle='--')
            
        # Adicionando rótulos e título
        plt.xlabel(coluna_x)
        plt.ylabel(data_y.name)
        plt.title(f'Dispersão {coluna_x} x {data_y.name}')
        plt.legend()
            
        # Salvando o gráfico
        figure_name = f'{coluna_x} x {data_y.name}.png'
        plt.savefig(os.path.join(path_save_folder, figure_name))
        plt.close()
        
    #plotando desempenho em campo
    grupo_0x = data_y.loc[clusters==0]
    grupo_1x = data_y.loc[clusters==1]
    
    
    grupo_0y = data_y.loc[clusters==0]
    grupo_1y = data_y.loc[clusters==1]
    
    # Criando o gráfico de dispersão
    plt.figure(figsize=(10, 6))
            
    plt.scatter(grupo_0x, grupo_0y, color='blue', label='Grupo 0')
    plt.scatter(grupo_1x, grupo_1y, color='red', label='Group 1')
        
    # Calculando a reta de regressão para o Grupo 0
    slope_0, intercept_0, _, _, _ = linregress(grupo_0x, grupo_0y)
    plt.plot(grupo_0x, slope_0 * grupo_0x + intercept_0, color='blue', linestyle='--')
            
    # Calculando a reta de regressão para o Grupo 1
    slope_1, intercept_1, _, _, _ = linregress(grupo_1x, grupo_1y)
    plt.plot(grupo_1x, slope_1 * grupo_1x + intercept_1, color='red', linestyle='--')
            
    # Adicionando rótulos e título
    plt.xlabel(data_y.name)
    plt.ylabel(data_y.name)
    plt.title(f'Dispersão {data_y.name} x {data_y.name}')
    plt.legend()
    
    # Salvando o gráfico
    figure_name = f'{data_y.name} x {data_y.name}.png'
    plt.savefig(os.path.join(path_save_folder, figure_name))
    plt.close()  
    
def gerar_pandas_profile(data, tentativa, path_save):
    # Cria a pasta 'profile' se não existir
    save_path = os.path.join(path_save, 'profile')
    os.makedirs(save_path, exist_ok=True)
    
    # Gera o relatório de perfil
    profile = ProfileReport(data, title=f'Pandas Profile Report - Tentativa {tentativa}', explorative=True)
    
    # Salva o relatório em HTML
    profile.to_file(os.path.join(save_path, f'{tentativa}_pandas_profile.html'))
    
def elbow_method(Y_campo, coluna):
    '''
    Método do cotovelo (Elbow Method):
    
    Executar o algoritmo K-means para diferentes valores de k (número de clusters), por exemplo, variando de 1 a 10.
    
    Para cada valor de k, calcular a soma das distâncias quadradas intra-cluster (inertia).
    
    Plotar inertia de para cada valor de cluster.
    
    Visualize o gráfico resultante.
    
    Identificar o ponto onde há uma "curva de cotovelo", ou seja, onde a adição de mais clusters não resulta em uma diminuição significativa na métrica. 
    Esse ponto é considerado o número ideal de clusters.
    '''
    
    # Lista para armazenar os valores de inÃ©rcia
    inertia = []
    
    data = Y_campo[:][coluna].to_numpy()
    data = data.reshape(-1, 1)
    
    # Experimente diferentes valores de k
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
    
    # Plotando o gráfico do método do cotovelo
    plt.plot(range(1, 11), inertia, marker='o')
    plt.xlabel('Número de clusters')
    plt.ylabel('Inércia')
    plt.title('Método do Cotovelo - {}'.format(coluna))
    plt.show()
    
def investigate_Elbow_method(colunas, Y_campo, print = False):
    if print == True:
        for coluna in colunas:
            plt.figure()
            #o 22 representa o primeiro e o segundo time
            elbow_method(Y_campo[:22], coluna)
            elbow_method(Y_campo[22:], coluna)
        

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

def plot_boxplot_and_normality_test(df, base_path, folder_name):
    """
    Plota os dados fornecidos e salva os gráficos no caminho especificado.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados.
    base_path (str): Caminho base onde os gráficos serão salvos.
    folder_name (str): Nome da pasta onde os gráficos serão salvos.
    """
    # Criar a pasta se não existir
    save_path = os.path.join(base_path, folder_name)
    os.makedirs(save_path, exist_ok=True)
    
    for column in df.columns:
        # Obter os dados da coluna
        data = df[column]
        
        # Realizar o teste de Shapiro-Wilk para normalidade
        stat, p_value = stats.shapiro(data)
        
        # Criar uma nova figura para cada coluna
        plt.figure(figsize=(10, 6))
        
        # Plotar o boxplot
        plt.boxplot(data, vert=False)
        plt.title(f'Coluna: {column}\nTeste de Normalidade (Shapiro-Wilk): p-value = {p_value:.3f}')
        plt.xlabel(column)
        
        # Adicionar o valor de p no gráfico
        plt.text(0.95, 0.95, f'p-value: {p_value:.3f}', transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        
        # Salvar o gráfico
        figure_path = os.path.join(save_path, f'{column}.png')
        plt.savefig(figure_path)
        plt.close()

def run(path_to_data, path_save, n_clusters_Desempenho_campo = 2, oversample = False, normal = False, metade = False):
    
    full_data = pd.read_csv(path_to_data)
       
    correlacao_pearson(full_data, print = False, save_path=os.path.join(path_save, 'correlacao\\corr_todos_dados.png'))
    
    #variavel preditora
    ###########para utilizar todos dados#############
    X_cognitivo = full_data.iloc[:, 12:]
    
    ##########selecionando somente funções cognitivas#############
    colunas_func_cog = ['Memory span', 'Acuracia Go', 'Acuracia nogo', 'Capacidade de rastreamento', 'Flexibilidade cognitiva (B-A)']
    X_cognitivo = full_data[colunas_func_cog]
    
    #plotando o boxplot e fazendo teste de normalidade nos dados de funções cognitivas
    #plot_boxplot_and_normality_test(X_cognitivo, path_save, 'boxplots')
    
    #normalizando dados
    scaler_cog = StandardScaler()
    X_cognitivo_norm = scaler_cog.fit_transform(X_cognitivo)
    
    #Selecionando colunas com dados de desempenho em campo
    Y_campo = full_data.iloc[:, :12]
    colunas_campo = ['gols_feitos', 'gols_sofridos', 'gols_companheiros', 'saldo_gols']
    
    #plotando o boxplot e fazendo teste de normalidade nos dados de desempenho em campo
    #plot_boxplot_and_normality_test(full_data[colunas_campo], path_save, 'boxplots')
    
    #Investigar em quantos clusters é correto dividir os dados utilizando o método do cotovelo (Elbow Method)
    investigate_Elbow_method(colunas_campo, Y_campo, print = False)
    
    #Transformando em pandas para rodar o elbow method
    X_cognitivo_pd = pd.DataFrame(X_cognitivo_norm)
    
    #Investigar em quantos clusters é correto dividir os dados utilizando o método do cotovelo (Elbow Method)
    colunas_cog = X_cognitivo_pd.columns
    investigate_Elbow_method(colunas_cog, X_cognitivo_pd, print = False)
    
    #Normalizando os dados e aplicar cluster para os dados do desempenho em campo
    Y_gols_feitos = kmeans_cluster_teams(Y_campo[['gols_feitos']], n_clusters_Desempenho_campo)
    Y_gols_sofridos = kmeans_cluster_teams(Y_campo[['gols_sofridos']], n_clusters_Desempenho_campo)
    Y_gols_companheiros = kmeans_cluster_teams(Y_campo[['gols_companheiros']], n_clusters_Desempenho_campo)
    Y_saldo_gols = kmeans_cluster_teams(Y_campo[['saldo_gols']], n_clusters_Desempenho_campo)
    
    #Salvando correlação do dados cognitivos com dados de campo
    cog_campo = pd.concat([full_data[colunas_func_cog], full_data[colunas_campo]], axis = 1)
    #cog_campo = cog_campo.rename(columns={'Memory span': 'Memória', 'Acuracia Go': 'Atenção',
     #                                     'Acuracia nogo': 'Impulsividade', 'Capacidade de rastreamento': 'Rastreamento',
      #                                    'Flexibilidade cognitiva (B-A)': 'Flexibilidade cognitiva', 'gols_feitos': 'Gols feitos',
       #                                   'gols_sofridos': 'Gols sofridos', 'gols_companheiros': 'Gols companheiros',
        #                                  'saldo_gols': 'Saldo de gols'})
    
    cog_campo = cog_campo.rename(columns={'Memory span': 'Memória de trabalho visuoespacial', 'Acuracia Go': 'Atenção sustentada',
                                          'Acuracia nogo': 'Impulsivity', 'Capacidade de rastreamento': 'Capacidade de rastreamento',
                                          'Flexibilidade cognitiva (B-A)': 'Cognitive flexibility', 'gols_feitos': 'Individual goals',
                                          'gols_sofridos': 'Cconceded goals', 'gols_companheiros': 'Goals by teammates',
                                          'saldo_gols': 'Net team goals'})
    
    correlacao_spearman(cog_campo, print = False, save_path=os.path.join(path_save, 'correlacao\\spearman_ing'))
    
    # Chamando a função para gerar o profile
    #gerar_pandas_profile(cog_campo, 'cog_campo', path_save)
    #gerar_pandas_profile(full_data, 'full_data', path_save)
    
    
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
            
        #salvando gráficos de dispersão
        dispersao_grupos(X_cognitivo, Y_campo['gols_feitos'], Y_gols_feitos['cluster'], path_save = path_save)
        dispersao_grupos(X_cognitivo, Y_campo['gols_sofridos'], Y_gols_sofridos['cluster'], path_save = path_save)
        dispersao_grupos(X_cognitivo, Y_campo['gols_companheiros'], Y_gols_companheiros['cluster'], path_save = path_save)
        dispersao_grupos(X_cognitivo, Y_campo['saldo_gols'], Y_saldo_gols['cluster'], path_save = path_save)
        
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
            
        #salvando gráficos de dispersão
        dispersao_grupos(X_cognitivo, Y_campo['gols_feitos'], Y_gols_feitos['cluster'], path_save = path_save)
        dispersao_grupos(X_cognitivo, Y_campo['gols_sofridos'], Y_gols_sofridos['cluster'], path_save = path_save)
        dispersao_grupos(X_cognitivo, Y_campo['gols_companheiros'], Y_gols_companheiros['cluster'], path_save = path_save)
        dispersao_grupos(X_cognitivo, Y_campo['saldo_gols'], Y_saldo_gols['cluster'], path_save = path_save)
            
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

        #salvando gráficos de dispersão
        dispersao_grupos(X_cognitivo, Y_campo['gols_feitos'], Y_gols_feitos['cluster'], path_save = path_save)
        dispersao_grupos(X_cognitivo, Y_campo['gols_sofridos'], Y_gols_sofridos['cluster'], path_save = path_save)
        dispersao_grupos(X_cognitivo, Y_campo['gols_companheiros'], Y_gols_companheiros['cluster'], path_save = path_save)
        dispersao_grupos(X_cognitivo, Y_campo['saldo_gols'], Y_saldo_gols['cluster'], path_save = path_save)

if __name__ == "__main__":
    
    run(path_to_data = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\dataset.csv', 
        path_save = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\data\\Normal',
        n_clusters_Desempenho_campo = 2, oversample = False, normal = True, metade = False)
    
    
    
    

    
    
    
    