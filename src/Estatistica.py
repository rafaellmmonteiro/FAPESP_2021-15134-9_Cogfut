import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import shapiro
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import MultiComparison
import os

def save_table_as_image(data, columns, title, save_path, footer_text=None):
    fig, ax = plt.subplots(figsize=(12, 2 + (len(data) * 0.3)))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=data.values, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Adjust layout to fit the title and the table
    plt.subplots_adjust(top=0.85)
    
    plt.suptitle(title, fontsize=14, y=0.98)  # Set the title above the table
    if footer_text:
        plt.figtext(0.5, 0.01, footer_text, wrap=True, horizontalalignment='center', fontsize=12)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def medias(data, tentativa, path_save):
    save_path = os.path.join(path_save, 'estatisticas')
    os.makedirs(save_path, exist_ok=True)
    
    mean_values = data.mean()
    std_values = data.std()
    max_mean = mean_values.max()
    max_mean_col = mean_values.idxmax()

    combined_df = pd.DataFrame({
        'Algoritmo': mean_values.index,
        'Média': mean_values.values,
        'Desvio Padrão': std_values.values
    })

    footer_text = f'Maior média: {max_mean:.3f} na coluna {max_mean_col}'
    save_table_as_image(combined_df, combined_df.columns, f'Média e Desvio Padrão - Tentativa {tentativa}', os.path.join(save_path, f'{tentativa}_media_desvio_padrao.png'), footer_text)

def teste_normalidade(data, tentativa, path_save):
    save_path = os.path.join(path_save, 'estatisticas')
    os.makedirs(save_path, exist_ok=True)
    
    normality_results = []
    for column in data.columns:
        stat, p = shapiro(data[column])
        normality_results.append([column, stat, p])
    
    normality_df = pd.DataFrame(normality_results, columns=['Algoritmo', 'Estatística', 'Valor p'])
    save_table_as_image(normality_df, normality_df.columns, f'Teste de Normalidade - Tentativa {tentativa}', os.path.join(save_path, f'{tentativa}_teste_normalidade.png'))

def anova(data, tentativa, path_save):
    save_path = os.path.join(path_save, 'estatisticas')
    os.makedirs(save_path, exist_ok=True)
    
    _, p = f_oneway(data['Naive'], data['Random forest'], data['KNN'], data['Logistica'], data['SVM'], data['Rede neural'], data['Xgboost'])
    
    anova_results = [['ANOVA', p]]
    anova_df = pd.DataFrame(anova_results, columns=['Teste', 'Valor p'])
    
    alpha = 0.05
    if p <= alpha:
        footer_text = 'Hipótese nula rejeitada. Dados são diferentes'
    else:
        footer_text = 'Hipótese alternativa rejeitada. Resultados são iguais'
    
    save_table_as_image(anova_df, anova_df.columns, f'Teste de ANOVA - Tentativa {tentativa}', os.path.join(save_path, f'{tentativa}_anova.png'), footer_text)

def post_hoc_tukey(data, tentativa, path_save):
    save_path = os.path.join(path_save, 'estatisticas')
    os.makedirs(save_path, exist_ok=True)
    
    data4tukey = {
        'accuracy': np.concatenate([data[col] for col in data.columns]),
        'algoritmo': np.concatenate([[col]*len(data) for col in data.columns])
    }
    compara_algoritmos = MultiComparison(data4tukey['accuracy'], data4tukey['algoritmo'])
    teste_estatistico = compara_algoritmos.tukeyhsd()
    
    tukey_df = pd.DataFrame(data=teste_estatistico.summary().data[1:], columns=teste_estatistico.summary().data[0])
    save_table_as_image(tukey_df, tukey_df.columns, f'Teste Post hoc Tukey - Tentativa {tentativa}', os.path.join(save_path, f'{tentativa}_post_hoc_tukey.png'))

def plot_best(gf_over, gs_over, gc_over, sg_over):
    # Cálculo das médias e desvios padrão das colunas
    gf_means = gf_over.mean()
    gs_means = gs_over.mean()
    gc_means = gc_over.mean()
    sg_means = sg_over.mean()
    
    gf_std = gf_over.std()
    gs_std = gs_over.std()
    gc_std = gc_over.std()
    sg_std = sg_over.std()

    # Encontrando a maior média de cada variável
    max_gf = gf_means.max()
    max_gs = gs_means.max()
    max_gc = gc_means.max()
    max_sg = sg_means.max()

    # Encontrando a coluna correspondente a cada maior média
    best_gf_col = gf_means.idxmax()
    best_gs_col = gs_means.idxmax()
    best_gc_col = gc_means.idxmax()
    best_sg_col = sg_means.idxmax()

    # Desvios padrão correspondentes
    std_gf = gf_std[best_gf_col]
    std_gs = gs_std[best_gs_col]
    std_gc = gc_std[best_gc_col]
    std_sg = sg_std[best_sg_col]

    # Valores e cores correspondentes
    values = [max_gf, max_gs, max_gc, max_sg]
    errors = [std_gf, std_gs, std_gc, std_sg]
    classifiers = [best_gf_col, best_gs_col, best_gc_col, best_sg_col]
    colors = ['skyblue', 'skyblue', 'salmon', 'gold']  # Cores para as barras

    # Configurações do gráfico
    x_labels = ['Gols feitos/rodada', 'Gols sofridos/rodada',
                'Gols companheiros/rodada', 'Saldo Gols/rodada']
    x = np.arange(len(x_labels))  # posições das barras

    # Criando a figura e os eixos
    fig, ax = plt.subplots(figsize=(10, 6))

    # Barras com valores e erros
    bars = ax.bar(x, values, color=colors, yerr=errors, capsize=5)

    # Títulos e rótulos
    ax.set_xlabel('Desempenho jogos reduzidos', fontsize=14)
    ax.set_ylabel('Média da acurácia', fontsize=14)
    ax.set_title('Acurácia de predição de desempenho em jogos reduzidos', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=14)

    # Adicionando rótulos de valores nas barras
    def adicionar_rotulos(barras, valores, desvios):
        for barra, valor, desvio in zip(barras, valores, desvios):
            altura = barra.get_height()
            ax.annotate(f'{altura:.2f}\n{valor}',
                        xy=(barra.get_x() + barra.get_width() / 2, altura + desvio),
                        xytext=(0, 1),  # 6 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize = 14)

    adicionar_rotulos(bars, classifiers, errors)

    # Ajustando layout
    fig.tight_layout()

    # Ajustando o limite superior do eixo y para adicionar espaço para o texto
    plt.ylim(0, max(values) * 1.15)
    
    # Mostrando o gráfico
    plt.show()

if __name__ == "__main__":
    
    ########################  Lendo arquivos  ########################   
    # Ler arquivos da pasta Oversample
    gf_over = pd.read_csv('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\resultados_ml\\Oversample\\gf_resultados.csv')
    gs_over = pd.read_csv('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\resultados_ml\\Oversample\\gs_resultados.csv')
    gc_over = pd.read_csv('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\resultados_ml\\Oversample\\gc_resultados.csv')
    sg_over = pd.read_csv('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\resultados_ml\\Oversample\\sg_resultados.csv')
    
    # Ler arquivos da pasta Normal
    gf_normal = pd.read_csv('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\resultados_ml\\Normal\\gf_resultados.csv')
    gs_normal = pd.read_csv('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\resultados_ml\\Normal\\gs_resultados.csv')
    gc_normal = pd.read_csv('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\resultados_ml\\Normal\\gc_resultados.csv')
    sg_normal = pd.read_csv('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\resultados_ml\\Normal\\sg_resultados.csv')
    
    # Ler arquivos da pasta Metade
    gf_metade = pd.read_csv('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\resultados_ml\\Metade\\gf_resultados.csv')
    gs_metade = pd.read_csv('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\resultados_ml\\Metade\\gs_resultados.csv')
    gc_metade = pd.read_csv('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\resultados_ml\\Metade\\gc_resultados.csv')
    sg_metade = pd.read_csv('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\resultados_ml\\Metade\\sg_resultados.csv')
    
    path_save_over = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\modelo_final\\Oversample'
    path_save_normal = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\modelo_final\\Normal'
    path_save_metade = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\modelo_final\\Metade'
    ################## Teste de normalidade #######################
    teste_normalidade(gf_over, 'gf_over', path_save_over)
    teste_normalidade(gs_over, 'gs_over', path_save_over)
    teste_normalidade(gc_over, 'gc_over', path_save_over)
    teste_normalidade(sg_over, 'sg_over', path_save_over)
    teste_normalidade(gf_normal, 'gf_normal', path_save_normal)
    teste_normalidade(gs_normal, 'gs_normal', path_save_normal)
    teste_normalidade(gc_normal, 'gc_normal', path_save_normal)
    teste_normalidade(sg_normal, 'sg_normal', path_save_normal)
    teste_normalidade(gf_metade, 'gf_metade', path_save_metade)
    teste_normalidade(gs_metade, 'gs_metade', path_save_metade)
    teste_normalidade(gc_metade, 'gc_metade', path_save_metade)
    teste_normalidade(sg_metade, 'sg_metade', path_save_metade)
    
    ################## Teste de anova #######################
    anova(gf_over, 'gf_over', path_save_over)
    anova(gs_over, 'gs_over', path_save_over)
    anova(gc_over, 'gc_over', path_save_over)
    anova(sg_over, 'sg_over', path_save_over)
    anova(gf_normal, 'gf_normal', path_save_normal)
    anova(gs_normal, 'gs_normal', path_save_normal)
    anova(gc_normal, 'gc_normal', path_save_normal)
    anova(sg_normal, 'sg_normal', path_save_normal)
    anova(gf_metade, 'gf_metade', path_save_metade)
    anova(gs_metade, 'gs_metade', path_save_metade)
    anova(gc_metade, 'gc_metade', path_save_metade)
    anova(sg_metade, 'sg_metade', path_save_metade)
    
    ################## Medias #######################
    medias(gf_over, 'gf_over', path_save_over)
    medias(gf_normal, 'gf_normal', path_save_normal)
    medias(gs_over, 'gs_over', path_save_over)
    medias(gs_normal, 'gs_normal', path_save_normal)
    medias(gc_over, 'gc_over', path_save_over)
    medias(gc_normal, 'gc_normal', path_save_normal)
    medias(sg_over, 'sg_over', path_save_over)
    medias(sg_normal, 'sg_normal', path_save_normal)
    medias(gf_metade, 'gf_metade', path_save_metade)
    medias(gs_metade, 'gs_metade', path_save_metade)
    medias(gc_metade, 'gc_metade', path_save_metade)
    medias(sg_metade, 'sg_metade', path_save_metade)
    
    ################## Teste de post_hoc_tukey #######################
    post_hoc_tukey(gf_over, 'gf_over', path_save_over)
    post_hoc_tukey(gs_over, 'gs_over', path_save_over)
    post_hoc_tukey(gc_over, 'gc_over', path_save_over)
    post_hoc_tukey(sg_over, 'sg_over', path_save_over)
    post_hoc_tukey(gf_normal, 'gf_normal', path_save_normal)
    post_hoc_tukey(gs_normal, 'gs_normal', path_save_normal)
    post_hoc_tukey(gc_normal, 'gc_normal', path_save_normal)
    post_hoc_tukey(sg_normal, 'sg_normal', path_save_normal)
    post_hoc_tukey(gf_metade, 'gf_metade', path_save_metade)
    post_hoc_tukey(gs_metade, 'gs_metade', path_save_metade)
    post_hoc_tukey(gc_metade, 'gc_metade', path_save_metade)
    post_hoc_tukey(sg_metade, 'sg_metade', path_save_metade)
    
    ################### Plotando melhores #########################
    plot_best(gf_over, gs_over, gc_over, sg_over)
    
    