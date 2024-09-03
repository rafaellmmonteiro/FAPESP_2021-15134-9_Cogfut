import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
'''
def plot_vertical_clustered_data(ax, df, data_col, cluster_col, y_label, title):
    """
    Plota um gráfico vertical de dados unidimensionais com espaçamento horizontal para evitar sobreposição,
    colorindo os pontos de acordo com os clusters.

    Parâmetros:
    ax (matplotlib.axes.Axes): Axes object to plot on.
    df (pd.DataFrame): DataFrame contendo os dados.
    data_col (str): Nome da coluna contendo os valores dos dados.
    cluster_col (str): Nome da coluna contendo os clusters.
    y_label (str): Rótulo para o eixo y.
    title (str): Título do gráfico.
    """
    # Extraindo dados e clusters do DataFrame
    data = df[data_col].values
    clusters = df[cluster_col].values

    # Cores para cada cluster
    colors = ['red' if cluster == 0 else 'blue' for cluster in clusters]
   
    # Deslocamento horizontal aleatório para cada ponto
    horizontal_offset = 0.02
    x_positions = np.random.uniform(-horizontal_offset, horizontal_offset, size=len(data))

    # Criar a plotagem
    scatter = ax.scatter(x_positions, data, c=colors, alpha=0.6)

    # Adicionar rótulos e título
    ax.set_xticks([])
    ax.set_xlabel('Spacement for visualization')
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Adicionar legenda
    unique_clusters = np.unique(clusters)
    handles = [plt.Line2D([0], [0], marker='o' if cluster == 0 else '+', color='w', markerfacecolor='red' if cluster == 0 else 'blue', markersize=10) for cluster in unique_clusters]
    labels = [f'Class {cluster}' for cluster in unique_clusters]
    ax.legend(handles, labels)
'''
def plot_vertical_clustered_data(ax, df, data_col, cluster_col, y_label, title):
    """
    Plota um gráfico vertical de dados unidimensionais com espaçamento horizontal para evitar sobreposição,
    colorindo os pontos de acordo com os clusters e usando diferentes marcadores para os clusters.

    Parâmetros:
    ax (matplotlib.axes.Axes): Objeto Axes para plotar.
    df (pd.DataFrame): DataFrame contendo os dados.
    data_col (str): Nome da coluna contendo os valores dos dados.
    cluster_col (str): Nome da coluna contendo os clusters.
    y_label (str): Rótulo para o eixo y.
    title (str): Título do gráfico.
    """
    # Extraindo dados e clusters do DataFrame
    data = df[data_col].values
    clusters = df[cluster_col].values

    # Deslocamento horizontal aleatório para cada ponto
    horizontal_offset = 0.02
    x_positions = np.random.uniform(-horizontal_offset, horizontal_offset, size=len(data))

    # Plotar cada cluster com diferentes marcadores e cores
    for cluster in np.unique(clusters):
        cluster_data = data[clusters == cluster]
        cluster_positions = x_positions[clusters == cluster]
        marker = 'o' if cluster == 0 else '>'
        color = 'red' if cluster == 0 else 'blue'
        ax.scatter(cluster_positions, cluster_data, c=color, marker=marker, alpha=0.6, label=f'Class {cluster}')

    # Adicionar rótulos e título
    ax.set_xticks([])
    ax.set_xlabel('Spacement for visualization')
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Adicionar legenda
    ax.legend()

# Caminhos dos arquivos
path_gf = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\data\\Normal\\gf_cluster.xlsx'
path_gs = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\data\\Normal\\gs_cluster.xlsx'
path_gc = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\data\\Normal\\gc_cluster.xlsx'
path_sg = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\data\\Normal\\sg_cluster.xlsx'

# Lendo os dados
gf_data = pd.read_excel(path_gf)
gs_data = pd.read_excel(path_gs)
gc_data = pd.read_excel(path_gc)
sg_data = pd.read_excel(path_sg)

# Criando a figura e os subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Usando a função para cada subplot
plot_vertical_clustered_data(axs[0, 0], gf_data, 'gols_feitos', 'cluster', 'Individual goals per game', 'Cluster from individual goals per game')
plot_vertical_clustered_data(axs[0, 1], gs_data, 'gols_sofridos', 'cluster', 'Conceded goals per game', 'Cluster from conceded goals per game')
plot_vertical_clustered_data(axs[1, 0], gc_data, 'gols_companheiros', 'cluster', 'Goals by teammates per game', 'Cluster from goals by teammates per game')
plot_vertical_clustered_data(axs[1, 1], sg_data, 'saldo_gols', 'cluster', 'Net team goals per game', 'Clusters from net team goals per game')

# Ajustando layout
plt.tight_layout()

# Mostrar a plotagem
plt.show()
