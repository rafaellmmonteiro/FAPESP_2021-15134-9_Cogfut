import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi

def plot_radar_with_std_multiple(dfs, group_col, var_cols, titles, overall_title):
    """
    Plota múltiplos gráficos de radar com a média e o desvio padrão de dois grupos para várias variáveis.

    Parâmetros:
    dfs (list of pd.DataFrame): Lista de DataFrames contendo os dados brutos.
    group_col (str): Nome da coluna que contém as informações dos grupos.
    var_cols (list): Lista com os nomes das colunas das variáveis.
    titles (list): Lista de títulos para cada gráfico.
    """
    num_plots = len(dfs)
    
    # Criando subplots com base no número de DataFrames
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), subplot_kw=dict(polar=True))
    axs = axs.ravel()  # Convertendo matriz 2x2 em lista para facilitar iteração

    # Iterando sobre cada DataFrame e título correspondente
    for i in range(num_plots):
        df = dfs[i]
        title = titles[i]

        # Calculando a média e o desvio padrão para cada grupo
        group_stats = df.groupby(group_col).agg(['mean', 'std'])
        
        # Inicializando listas para médias e desvios padrão dos grupos
        group1_mean = group_stats.loc[0, (var_cols, 'mean')].values
        group1_std = group_stats.loc[0, (var_cols, 'std')].values
        group2_mean = group_stats.loc[1, (var_cols, 'mean')].values
        group2_std = group_stats.loc[1, (var_cols, 'std')].values
        
        # Ângulos para o gráfico de radar
        num_vars = len(var_cols)
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]

        # Adicionando o gráfico atual ao subplot correspondente
        ax = axs[i]
        
        # Função para adicionar uma linha com a média e a área com o desvio padrão
        def add_group_to_radar(ax, angles, means, std_devs, label, color):
            means = means.tolist() + means.tolist()[:1]
            std_devs = std_devs.tolist() + std_devs.tolist()[:1]

            lower_bound = [m - s for m, s in zip(means, std_devs)]
            upper_bound = [m + s for m, s in zip(means, std_devs)]

            ax.plot(angles, means, linewidth=2, linestyle='solid', label=label, color=color)
            ax.fill_between(angles, lower_bound, upper_bound, color=color, alpha=0.2)

        # Adicionar grupo 1
        add_group_to_radar(ax, angles, group1_mean, group1_std, 'Cluster 0', 'red')

        # Adicionar grupo 2
        add_group_to_radar(ax, angles, group2_mean, group2_std, 'Cluster 1', 'blue')

        # Adicionar rótulos
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(var_cols, color='grey', size=12)
        ax.set_rlabel_position(0)
        ax.set_yticks([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
        ax.set_yticklabels(["-2.5", "-1.5", "-0.5", "0.5", "1.5", "2.5"], color="grey", size=7)
        ax.set_ylim(-2.5, 2.5)

        # Adicionar título e legenda
        ax.set_title(title, size=15, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
    # Adicionar título geral à figura
    plt.suptitle(overall_title, size=20)

    # Ajustando layout
    plt.tight_layout()

    # Mostrar os gráficos
    plt.show()

# Caminhos dos arquivos
path_gf = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\data\\Normal\\gf_cluster_cog.xlsx'
path_gs = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\data\\Normal\\gs_cluster_cog.xlsx'
path_gc = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\data\\Normal\\gc_cluster_cog.xlsx'
path_sg = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\data\\Normal\\sg_cluster_cog.xlsx'

# Lendo os dados
gf_data = pd.read_excel(path_gf)
gs_data = pd.read_excel(path_gs)
gc_data = pd.read_excel(path_gc)
sg_data = pd.read_excel(path_sg)

# Usando a função para plotar múltiplos gráficos de radar
plot_radar_with_std_multiple([gf_data, gs_data, gc_data, sg_data],
                             'cluster',
                             ['Memória', 'Atenção', 'Impulsividade', 'Rastreamento', 'Flexibilidade cognitiva'],
                             ['Gols feitos por rodada', 'Gols sofridos por rodada', 'Gols feitos por companheiros por rodada', 'Saldo de gols por rodada'],
                             'Comparação das funções cognitivas separadas em clusters')


