import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np
import os
import altair as at
from IPython.display import display, HTML

def plotar_dados_cruzados(corsi_data, gonogo_data, mot_data, ptrails_data, jogos_reduzidos_data, 
                          plot_corsi = False, plot_gonogo = False, plot_mot = False, plot_ptrails = False):
   
    #selecionar apenas dados cognitivos que tenham dados de jogos reduzidos
    jr_jogadores = jogos_reduzidos_data.dropna(subset=['gols_feitos'])
    
    corsi_jog = corsi_data[corsi_data['Numero'].isin(jr_jogadores['Numero'])].sort_values(by='Numero')
    gonogo_jog = gonogo_data[gonogo_data['Numero'].isin(jr_jogadores['Numero'])].sort_values(by='Numero')
    mot_jog = mot_data[mot_data['Numero'].isin(jr_jogadores['Numero'])].sort_values(by='Numero')
    ptrails_jog = ptrails_data[ptrails_data['Numero'].isin(jr_jogadores['Numero'])].sort_values(by='Numero')
    jr_jogadores = jogos_reduzidos_data[jogos_reduzidos_data['Numero'].isin(ptrails_jog['Numero'])].sort_values(by='Numero')
    
    #plotar dados corsi
    if (plot_corsi == True):
        # Iterar sobre as colunas de jr_jogadores
        for col_jr in jr_jogadores.columns[3:10]:
    
            # Cria uma figura com 1 linha e 4 colunas
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))
            
            # Itera sobre as colunas de corsi_jog
            for i, col_corsi in enumerate(corsi_jog.columns[2:]):
                
                # Criar um novo gráfico de dispersão com linha de tendência usando Seaborn regplot
                sns.regplot(x=col_jr, y=col_corsi, data=pd.concat([jr_jogadores[col_jr], corsi_jog[col_corsi]], axis=1), ax=axs[i])
               
                # Calcula a correlação linear de Pearson
                correlacao, _ = pearsonr(jr_jogadores[col_jr], corsi_jog[col_corsi])
            
                # Adiciona o valor da correlação ao título
                axs[i].set_title(f'{col_jr} x {col_corsi}\nCorrelação: {correlacao:.2f}')
                axs[i].set_xlabel(col_jr)
                axs[i].set_ylabel(col_corsi)
    
            # Ajusta o layout para evitar sobreposições
            plt.tight_layout()
            
    #plotar dados gonogo
    if (plot_gonogo == True):
        # Iterar sobre as colunas de jr_jogadores
        for col_jr in jr_jogadores.columns[3:10]:
    
            # Cria uma figura com 2 linhas e 4 colunas
            fig, axs = plt.subplots(2, 4, figsize=(16, 8))
            
            # Itera sobre as colunas de corsi_jog
            for i, col_gonogo in enumerate(gonogo_jog.columns[2:]):
                
                # Obtém o subplot atual
                ax = axs[i // 4, i % 4]
                
                # Criar um novo gráfico de dispersão com linha de tendência usando Seaborn regplot
                sns.regplot(x=col_jr, y=col_gonogo, data=pd.concat([jr_jogadores[col_jr], gonogo_jog[col_gonogo]], axis=1), ax=ax)
            
                # Calcula a correlação linear de Pearson
                correlacao, _ = pearsonr(jr_jogadores[col_jr], gonogo_jog[col_gonogo])
            
                # Adiciona o valor da correlação ao título
                ax.set_title(f'{col_jr} x {col_gonogo}\nCorrelação: {correlacao:.2f}')
                ax.set_xlabel(col_jr)
                ax.set_ylabel(col_gonogo)
    
            # Ajusta o layout para evitar sobreposições
            plt.tight_layout()
    
    #plotar dados mot
    if (plot_mot == True):
        # Iterar sobre as colunas de jr_jogadores
        for col_jr in jr_jogadores.columns[3:10]:
    
            # Cria uma figura com 1 linha e 4 colunas
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            
            # Itera sobre as colunas de corsi_jog
            for i, col_mot in enumerate(mot_jog.columns[2:]):
                
                # Criar um novo gráfico de dispersão com linha de tendência usando Seaborn regplot
                sns.regplot(x=col_jr, y=col_mot, data=pd.concat([jr_jogadores[col_jr], mot_jog[col_mot]], axis=1), ax=axs[i])
               
                # Calcula a correlação linear de Pearson
                correlacao, _ = pearsonr(jr_jogadores[col_jr], mot_jog[col_mot])
            
                # Adiciona o valor da correlação ao título
                axs[i].set_title(f'{col_jr} x {col_mot}\nCorrelação: {correlacao:.2f}')
                axs[i].set_xlabel(col_jr)
                axs[i].set_ylabel(col_mot)
    
            # Ajusta o layout para evitar sobreposições
            plt.tight_layout()
            
    #plotar dados ptrails
    if (plot_ptrails == True):
        # Iterar sobre as colunas de jr_jogadores
        for col_jr in jr_jogadores.columns[3:10]:
    
            # Cria uma figura com 2 linhas e 4 colunas
            fig, axs = plt.subplots(2, 3, figsize=(16, 8))
            
            # Itera sobre as colunas de corsi_jog
            for i, col_ptrails in enumerate(ptrails_jog.columns[2:]):
                
                # Obtém o subplot atual
                ax = axs[i // 3, i % 3]
                
                # Criar um novo gráfico de dispersão com linha de tendência usando Seaborn regplot
                sns.regplot(x=col_jr, y=col_ptrails, data=pd.concat([jr_jogadores[col_jr], ptrails_jog[col_ptrails]], axis=1), ax=ax)
            
                # Calcula a correlação linear de Pearson
                correlacao, _ = pearsonr(jr_jogadores[col_jr], ptrails_jog[col_ptrails])
            
                # Adiciona o valor da correlação ao título
                ax.set_title(f'{col_jr} x {col_ptrails}\nCorrelação: {correlacao:.2f}')
                ax.set_xlabel(col_jr)
                ax.set_ylabel(col_ptrails)
    
            # Ajusta o layout para evitar sobreposições
            plt.tight_layout()
            
def radar_boxplot(data_group1, data_group2, labels=('Média', 'Desempenho individual')):
    # Verifica se os grupos têm o mesmo número de variáveis
    if len(data_group1) != len(data_group2):
        raise ValueError("Os grupos devem ter o mesmo número de variáveis")

    # Nomes das variáveis (pontas) no gráfico
    variaveis = [f'Variável {i+1}' for i in range(len(data_group1))]
    variaveis[0] = 'Memória'
    variaveis[1] = 'Atenção'
    variaveis[2] = 'Impulsividade'
    variaveis[3] = 'Rastreamento'
    variaveis[4] = 'Flexibilidade cognitiva'

    # Número de variáveis
    num_variaveis = len(variaveis)

    # Angulos para as variáveis
    angulos = np.linspace(0, 2 * np.pi, num_variaveis, endpoint=False).tolist()

    # Adicione o primeiro elemento no final para que o gráfico seja fechado
    angulos += angulos[:1]

    # Plote o gráfico de radar
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Adicione um ponto adicional para fechar o polígono
    data_group1 = np.append(data_group1, data_group1[0])
    data_group2 = np.append(data_group2, data_group2[0])

    # Plote os dados do Grupo 1
    ax.plot(angulos, data_group1, label=labels[0], alpha=0.5)

    # Plote os dados do Grupo 2
    ax.plot(angulos, data_group2, label=labels[1], alpha=0.5, color = 'red')

    # Adicione legendas e título
    ax.set_thetagrids(np.degrees(angulos[:-1]), variaveis)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Desempenho cognitivo')

    # Exiba o gráfico
    plt.show()

def plot_radar(corsi_data, gonogo_data, mot_data, ptrails_data, numero):
    
    # Normalizar os dados de cada DataFrame apenas para as colunas desejadas
    corsi_data_normalized = corsi_data.copy()
    gonogo_data_normalized = gonogo_data.copy()
    mot_data_normalized = mot_data.copy()
    ptrails_data_normalized = ptrails_data.copy()
    
    corsi_data_normalized['Memory span'] = (corsi_data['Memory span'] - corsi_data['Memory span'].min()) / (corsi_data['Memory span'].max() - corsi_data['Memory span'].min())
    gonogo_data_normalized['Acuracia Go'] = (gonogo_data['Acuracia Go'] - gonogo_data['Acuracia Go'].min()) / (gonogo_data['Acuracia Go'].max() - gonogo_data['Acuracia Go'].min())
    gonogo_data_normalized['Acuracia nogo'] = (gonogo_data['Acuracia nogo'] - gonogo_data['Acuracia nogo'].min()) / (gonogo_data['Acuracia nogo'].max() - gonogo_data['Acuracia nogo'].min())
    mot_data_normalized['Capacidade de rastreamento'] = (mot_data['Capacidade de rastreamento'] - mot_data['Capacidade de rastreamento'].min()) / (mot_data['Capacidade de rastreamento'].max() - mot_data['Capacidade de rastreamento'].min())
    ptrails_data_normalized['Flexibilidade cognitiva (B-A)'] = 1 - (ptrails_data['Flexibilidade cognitiva (B-A)'] - ptrails_data['Flexibilidade cognitiva (B-A)'].min()) / (ptrails_data['Flexibilidade cognitiva (B-A)'].max() - ptrails_data['Flexibilidade cognitiva (B-A)'].min())
    
    #medias dos testes cognitivos
    medias = [
        corsi_data_normalized['Memory span'].mean(),
        gonogo_data_normalized['Acuracia Go'].mean(),
        gonogo_data_normalized['Acuracia nogo'].mean(),
        mot_data_normalized['Capacidade de rastreamento'].mean(),
        ptrails_data_normalized['Flexibilidade cognitiva (B-A)'].mean()
    ]
    
    #sujeito
    sujeitos = [
            corsi_data_normalized.loc[corsi_data['Numero'] == numero, 'Memory span'],
            gonogo_data_normalized.loc[gonogo_data['Numero'] == numero, 'Acuracia Go'],
            gonogo_data_normalized.loc[gonogo_data['Numero'] == numero, 'Acuracia nogo'],
            mot_data_normalized.loc[mot_data['Numero'] == numero, 'Capacidade de rastreamento'],
            ptrails_data_normalized.loc[ptrails_data['Numero'] == numero, 'Flexibilidade cognitiva (B-A)']
        ]

    
    radar_boxplot(medias, sujeitos)
    
def save_radar_data(corsi_data, gonogo_data, mot_data, ptrails_data, lista_sujeitos, path_to_save):
    # Filtrar os dados para os sujeitos presentes na lista
    corsi_data_filtered = corsi_data[corsi_data['Numero'].isin(lista_sujeitos)]
    gonogo_data_filtered = gonogo_data[gonogo_data['Numero'].isin(lista_sujeitos)]
    mot_data_filtered = mot_data[mot_data['Numero'].isin(lista_sujeitos)]
    ptrails_data_filtered = ptrails_data[ptrails_data['Numero'].isin(lista_sujeitos)]
    
    # Normalizar os dados para os sujeitos filtrados
    corsi_data_normalized = corsi_data_filtered.copy()
    gonogo_data_normalized = gonogo_data_filtered.copy()
    mot_data_normalized = mot_data_filtered.copy()
    ptrails_data_normalized = ptrails_data_filtered.copy()
    
    corsi_data_normalized['Memory span'] = (corsi_data_filtered['Memory span'] - corsi_data_filtered['Memory span'].min()) / (corsi_data_filtered['Memory span'].max() - corsi_data_filtered['Memory span'].min())
    gonogo_data_normalized['Acuracia Go'] = (gonogo_data_filtered['Acuracia Go'] - gonogo_data_filtered['Acuracia Go'].min()) / (gonogo_data_filtered['Acuracia Go'].max() - gonogo_data_filtered['Acuracia Go'].min())
    gonogo_data_normalized['Acuracia nogo'] = (gonogo_data_filtered['Acuracia nogo'] - gonogo_data_filtered['Acuracia nogo'].min()) / (gonogo_data_filtered['Acuracia nogo'].max() - gonogo_data_filtered['Acuracia nogo'].min())
    mot_data_normalized['Capacidade de rastreamento'] = (mot_data_filtered['Capacidade de rastreamento'] - mot_data_filtered['Capacidade de rastreamento'].min()) / (mot_data_filtered['Capacidade de rastreamento'].max() - mot_data_filtered['Capacidade de rastreamento'].min())
    ptrails_data_normalized['Flexibilidade cognitiva (B-A)'] = 1 - (ptrails_data_filtered['Flexibilidade cognitiva (B-A)'] - ptrails_data_filtered['Flexibilidade cognitiva (B-A)'].min()) / (ptrails_data_filtered['Flexibilidade cognitiva (B-A)'].max() - ptrails_data_filtered['Flexibilidade cognitiva (B-A)'].min())
    
    # Criar um DataFrame com os dados filtrados e normalizados
    radar_data = pd.DataFrame({
        'Sujeito': lista_sujeitos,
        'Memory span': corsi_data_normalized['Memory span'],
        'Acuracia Go': gonogo_data_normalized['Acuracia Go'],
        'Acuracia nogo': gonogo_data_normalized['Acuracia nogo'],
        'Capacidade de rastreamento': mot_data_normalized['Capacidade de rastreamento'],
        'Flexibilidade cognitiva (B-A)': ptrails_data_normalized['Flexibilidade cognitiva (B-A)']
    })
    
    radar_data['Capacidade de rastreamento'] = radar_data['Sujeito'].map(mot_data_normalized.set_index('Numero')['Capacidade de rastreamento'])
    
    path_cog = os.path.join(path_to_save, 'dados_cognitivos_data.csv')
    
    # Salvar o DataFrame como um arquivo CSV
    radar_data.to_csv(path_cog, index=False)
    
def correlacao_cruzada(corsi_data, gonogo_data, mot_data, ptrails_data, jogos_reduzidos_data_media, jogos_reduzidos_data_total, time):
    
    #selecionar apenas dados cognitivos que tenham dados de jogos reduzidos
    jr_jogadores_media = jogos_reduzidos_data_media.dropna(subset=['gols_feitos'])
    jr_jogadores_media = jr_jogadores_media[jr_jogadores_media['rodadas'] > 20]

    
    corsi_jog = corsi_data[corsi_data['Numero'].isin(jr_jogadores_media['Numero'])].sort_values(by='Numero')
    gonogo_jog = gonogo_data[gonogo_data['Numero'].isin(jr_jogadores_media['Numero'])].sort_values(by='Numero')
    mot_jog = mot_data[mot_data['Numero'].isin(jr_jogadores_media['Numero'])].sort_values(by='Numero')
    ptrails_jog = ptrails_data[ptrails_data['Numero'].isin(jr_jogadores_media['Numero'])].sort_values(by='Numero')
    jr_jogadores_media = jogos_reduzidos_data_media[jogos_reduzidos_data_media['Numero'].isin(ptrails_jog['Numero'])].sort_values(by='Numero')
    jr_jogadores_total = jogos_reduzidos_data_total[jogos_reduzidos_data_total['Numero'].isin(jr_jogadores_media['Numero'])].sort_values(by='Numero')
    
    #selecionando dados para fazer correlação
    jr_jogadores_total = jr_jogadores_total[['gols_feitos','gols_sofridos', 'gols_companheiros', 'saldo_gols']].reset_index(drop = True)
    jr_jogadores_media = jr_jogadores_media.drop(columns=['Unnamed: 0', 'Nome_do_jogador']).reset_index(drop = True)
    corsi_jog = corsi_jog.drop(columns=['Unnamed: 0']).reset_index(drop = True)
    gonogo_jog = gonogo_jog.drop(columns=['Unnamed: 0']).reset_index(drop = True)
    mot_jog = mot_jog.drop(columns=['Unnamed: 0']).reset_index(drop = True)
    ptrails_jog = ptrails_jog.drop(columns=['Unnamed: 0']).reset_index(drop = True)
    
    #renomeando colunas de jogos reduzidos total
    # Criar um dicionário de mapeamento para renomear as colunas
    rename_mapping = {col: col + '_total' for col in jr_jogadores_total.columns}
    
    # Renomear as colunas usando o método rename()
    jr_jogadores_total.rename(columns=rename_mapping, inplace=True)

    
    #fazendo arquivo para calcular correlações
    data4corr = pd.concat([jr_jogadores_media, jr_jogadores_total, corsi_jog, gonogo_jog, mot_jog, ptrails_jog], axis=1)
    
    #salvando todos dados tabelados
    data4corr.to_csv(f'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\full_data_{time}.csv')
    
    cor_data = (data4corr.drop(columns=['Numero'])
              .corr().stack()
              .reset_index()     # The stacking results in an index on the correlation values, we need the index as normal columns for Altair
              .rename(columns={0: 'correlation', 'level_0': 'variable', 'level_1': 'variable2'}))
    cor_data['correlation_label'] = cor_data['correlation'].map('{:.2f}'.format)  # Round to 2 decimal
    cor_data.head()
    
    base = at.Chart(cor_data).encode(
    x='variable2:O',
    y='variable:O'    
    ).properties(
    width=750,  # Defina a largura do gráfico
    height=750  # Defina a altura do gráfico
    )
    
    # Text layer with correlation labels
    # Colors are for easier readability
    text = base.mark_text().encode(
        text='correlation_label',
        color=at.condition(
            at.datum.correlation > 0.5, 
            at.value('white'),
            at.value('black')
        )
    )
    
    # The correlation heatmap itself
    cor_plot = base.mark_rect().encode(
        color='correlation:Q'
    )
    
    chart = cor_plot + text
    
    # Salvar o gráfico como um arquivo HTML
    chart.save(f'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\grafico_correlacao_{time}.html')
    
    return data4corr

def correlacao_cruzada_full(data4corr):
    
    #plotando gráfico de correlação cruzada
    cor_data = (data4corr.drop(columns=['Numero'])
              .corr().stack()
              .reset_index()     # The stacking results in an index on the correlation values, we need the index as normal columns for Altair
              .rename(columns={0: 'correlation', 'level_0': 'variable', 'level_1': 'variable2'}))
    cor_data['correlation_label'] = cor_data['correlation'].map('{:.2f}'.format)  # Round to 2 decimal
    cor_data.head()
    
    base = at.Chart(cor_data).encode(
    x='variable2:O',
    y='variable:O'    
    ).properties(
    width=750,  # Defina a largura do gráfico
    height=750  # Defina a altura do gráfico
    )
    
    # Text layer with correlation labels
    # Colors are for easier readability
    text = base.mark_text().encode(
        text='correlation_label',
        color=at.condition(
            at.datum.correlation > 0.5, 
            at.value('white'),
            at.value('black')
        )
    )
    
    # The correlation heatmap itself
    cor_plot = base.mark_rect().encode(
        color='correlation:Q'
    )
    
    chart = cor_plot + text
    
    # Salvar o gráfico como um arquivo HTML
    chart.save(f'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\grafico_correlacao_full.html')
    
    return data4corr
    
if __name__ == "__main__":
    
    # Criar um DataFrame vazio para armazenar os resultados combinados
    resultados_finais = pd.DataFrame()

    path_to_data = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\data\\results'
    
    for time in os.listdir(path_to_data):
        # Verificar se o item é a pasta 'results'
        if time == 'dados_cognitivos_data.csv':
            continue  # Ignorar a pasta 'results' e continuar com o próximo item
        
        path_to_corsi = (os.path.join(path_to_data, time, 'corsi_data.csv'))
        
        path_to_gonogo = (os.path.join(path_to_data, time, 'gonogo_data.csv')) 
        
        path_to_ptrails = (os.path.join(path_to_data, time, 'ptrails_data.csv'))
        
        path_to_mot = (os.path.join(path_to_data, time, 'mot_data.csv'))
        
        path_to_jogos_reduzidos_media = (os.path.join(path_to_data, time, 'Media_performance_jogadores.csv'))
        
        path_to_jogos_reduzidos_total = (os.path.join(path_to_data, time, 'Total_performance_jogadores.csv'))
        
        path_to_save = path_to_data
    
        corsi_data = pd.read_csv(path_to_corsi)
        
        gonogo_data = pd.read_csv(path_to_gonogo)
        
        mot_data = pd.read_csv(path_to_mot)
        
        ptrails_data = pd.read_csv(path_to_ptrails)
        
        jogos_reduzidos_data_media = pd.read_csv(path_to_jogos_reduzidos_media)
        
        jogos_reduzidos_data_total = pd.read_csv(path_to_jogos_reduzidos_total)
        
        #plot_radar(corsi_data, gonogo_data, mot_data, ptrails_data, 1)
        
        #plota dados cruzados
        plotar_dados_cruzados(corsi_data, gonogo_data, mot_data, ptrails_data, jogos_reduzidos_data_media, 
                                  plot_corsi=  False, plot_gonogo = False, plot_mot = False, plot_ptrails = False)
            
        
        #realiza a correlação cruzada dos dados
        full_data = correlacao_cruzada(corsi_data, gonogo_data, mot_data, ptrails_data, jogos_reduzidos_data_media,
                                                    jogos_reduzidos_data_total, time)
        
        #salvar dados normalizados utilizados para fazer o plot_radar
        save_radar_data(corsi_data, gonogo_data, mot_data, ptrails_data, corsi_data['Numero'], path_to_save)
        
        # Adiciona o DataFrame full_data ao DataFrame resultados_finais
        resultados_finais = pd.concat([resultados_finais, full_data])
        
    #Salva todos os resultados juntos
    resultados_finais.to_csv('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\full_data.csv')
    
    #retirando colunas desnecessárias
    resultados_finais_corr = resultados_finais.drop(columns=['Numero'])
    
    #Corrigindo outliers
    resultados_finais_corr = resultados_finais_corr.reset_index(drop = True)
    lista_para_corrigir = ['Último item com tentativa completa', 'Total score', 'Total tentativas correctas', 'Memory span']
    for correct in lista_para_corrigir:
        resultados_finais_corr.loc[resultados_finais_corr[correct] < 2, correct] = resultados_finais_corr[correct][resultados_finais_corr[correct] > 2].mean()
    
    #Salva todos os resultados juntos
    resultados_finais_corr.to_csv('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\dataset.csv', index = False)
    
    #Rodando correlação cruzada em todos os dados e salvando html
    grafico_correlacao = correlacao_cruzada_full(resultados_finais)
       