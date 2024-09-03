import pandas as pd
import os

def read_corsi(path_to_corsi):
    
    resultados_corsi = pd.DataFrame(columns=['Numero', 'Último item com tentativa completa', 'Total score', 
                                             'Total tentativas correctas', 'Memory span'])
    # Listar todos os diretórios no caminho_base
    diretorios = [d for d in os.listdir(path_to_corsi) if os.path.isdir(os.path.join(path_to_corsi, d))]
    
    # Iterar sobre cada diretório
    for diretorio in diretorios:
        # Construir o caminho completo para o diretório atual
        caminho_diretorio = os.path.join(path_to_corsi, diretorio)
    
        # Listar todos os arquivos no diretório atual
        arquivos = [f for f in os.listdir(caminho_diretorio) if os.path.isfile(os.path.join(caminho_diretorio, f))]
    
        # Iterar sobre cada arquivo
        for arquivo in arquivos:
            # Verificar se o arquivo é um arquivo CSV e contém "summary" no nome
            if arquivo.endswith('.csv') and 'summary' in arquivo.lower():
                caminho_arquivo = os.path.join(caminho_diretorio, arquivo)
    
                # Ler o arquivo CSV usando pandas
                resultado_corsi = pd.read_csv(caminho_arquivo, sep = ':')
                resultado_corsi = resultado_corsi[3:7]
                resultado_corsi = resultado_corsi.transpose().reset_index()
                resultado_corsi = resultado_corsi.rename(columns={'index': 'Numero'})
                resultado_corsi['Numero'] = diretorio
                resultado_corsi = resultado_corsi.apply(pd.to_numeric)
                resultados_corsi = pd.concat([resultados_corsi, resultado_corsi])
                
    return resultados_corsi

def read_gonogo(path_to_gonogo):
    
    resultados_gonogo = pd.DataFrame(columns=['Numero', 'Total corretos', 'Total erros', 'Acuracia Media', 
                                             'Erros media', 'Acuracia Go', 'Acuracia nogo', 'Tempo resposta Go', 'Tempo resposta nogo'])
    linha_gonogo = resultados_gonogo.copy()
    linha_gonogo.loc[0] = 0
    
    # Listar todos os diretórios no caminho_base
    diretorios = [d for d in os.listdir(path_to_gonogo) if os.path.isdir(os.path.join(path_to_gonogo, d))]
    
    # Iterar sobre cada diretório
    for diretorio in diretorios:
        # Construir o caminho completo para o diretório atual
        caminho_diretorio = os.path.join(path_to_gonogo, diretorio)
    
        # Listar todos os arquivos no diretório atual
        arquivos = [f for f in os.listdir(caminho_diretorio) if os.path.isfile(os.path.join(caminho_diretorio, f))]
    
        # Iterar sobre cada arquivo
        for arquivo in arquivos:
            # Verificar se o arquivo é um arquivo txt
            if arquivo.endswith('.txt'):
                caminho_arquivo = os.path.join(caminho_diretorio, arquivo)
    
                # Ler o arquivo CSV usando pandas
                resultado_gonogo = pd.read_csv(caminho_arquivo)
                
                #extraindo dados sumario
                summary_gonogo = resultado_gonogo[11:15].reset_index()
                summary_gonogo_transposed = summary_gonogo['index'].str.split(':', expand=True).transpose()
                
                #extraindo dados round 1
                #acuracia
                round_1_ac_gonogo = resultado_gonogo[19:21].reset_index()
                round_1_ac_gonogo_sep = round_1_ac_gonogo['index'].str.extractall(r'(\S+)')
                
                #tempo de resposta
                round_1_rt_gonogo = resultado_gonogo[32:34].reset_index()
                round_1_rt_gonogo_sep = round_1_rt_gonogo['index'].str.extractall(r'(\S+)')
                
                #extraindo dados round 2
                #acuracia
                round_2_ac_gonogo = resultado_gonogo[25:27].reset_index()
                round_2_ac_gonogo_sep = round_2_ac_gonogo['index'].str.extractall(r'(\S+)')
                
                #tempo de resposta
                round_2_rt_gonogo = resultado_gonogo[39:41].reset_index()
                round_2_rt_gonogo_sep = round_2_rt_gonogo['index'].str.extractall(r'(\S+)')
                
                #####organizando as variaveis nas linhas para adicionar aos resultados#######
                #numero
                linha_gonogo.iloc[0, 0] = pd.to_numeric(diretorio)
                
                #variaveis do sumário
                linha_gonogo.iloc[0, 1:5] = pd.to_numeric(summary_gonogo_transposed.iloc[1, :4])
                
                #acurácia Go
                linha_gonogo.iloc[0, 5] = (pd.to_numeric(round_1_ac_gonogo_sep.iloc[3]) + 
                                           pd.to_numeric(round_2_ac_gonogo_sep.iloc[8]))/2 
                
                #acurácia noGo
                linha_gonogo.iloc[0, 6] = (pd.to_numeric(round_1_ac_gonogo_sep.iloc[8]) + 
                                           pd.to_numeric(round_2_ac_gonogo_sep.iloc[3]))/2 
                
                #Tempo resposta Go
                linha_gonogo.iloc[0, 7] = (pd.to_numeric(round_1_rt_gonogo_sep.iloc[3]) + 
                                           pd.to_numeric(round_2_rt_gonogo_sep.iloc[8]))/2 
                
                #Tempo resposta noGo
                linha_gonogo.iloc[0, 8] = (pd.to_numeric(round_1_rt_gonogo_sep.iloc[8]) + 
                                           pd.to_numeric(round_2_rt_gonogo_sep.iloc[3]))/2 
                
                resultados_gonogo = pd.concat([resultados_gonogo, linha_gonogo])
                
    return resultados_gonogo

def read_mot(path_to_mot):
    
    #definindo nome das colunas arquivo de saida
    nome_colunas = ['Numero', 'Media de acertos', 'Capacidade de rastreamento']
    resultados_mot = pd.DataFrame(columns=nome_colunas)

    
    # Listar todos os diretórios no caminho_base
    arquivos = os.listdir(path_to_mot)
    
    # Iterar sobre cada diretório
    for arquivo in arquivos:
        
        if arquivo.endswith('.txt') and 'short' in arquivo.lower():
            # Construir o caminho completo para o diretório atual
            caminho_arquivo = os.path.join(path_to_mot, arquivo)
        
            # Ler o arquivo CSV usando pandas
            resultado_mot = pd.read_csv(caminho_arquivo, sep = ';')
            resultado_mot.columns = nome_colunas
            
            #escrevendo resultados 
            resultados_mot = pd.concat([resultados_mot, resultado_mot])
            
    return resultados_mot

def read_ptrails(path_to_ptrails):
    
    resultados_ptrails = pd.DataFrame(columns=['Numero', 'Melhor tempo total A', 'Melhor tempo total B', 
                                             'Flexibilidade cognitiva (B-A)', 'Acuracia media A', 'Acuracia media B'])
    linha_ptrails = resultados_ptrails.copy()
    linha_ptrails.loc[0] = 0
    
    # Listar todos os diretórios no caminho_base
    diretorios = [d for d in os.listdir(path_to_corsi) if os.path.isdir(os.path.join(path_to_ptrails, d))]
    
    # Iterar sobre cada diretório
    for diretorio in diretorios:
        # Construir o caminho completo para o diretório atual
        caminho_diretorio = os.path.join(path_to_ptrails, diretorio)
    
        # Listar todos os arquivos no diretório atual
        arquivos = [f for f in os.listdir(caminho_diretorio) if os.path.isfile(os.path.join(caminho_diretorio, f))]
    
        # Iterar sobre cada arquivo
        for arquivo in arquivos:
            # Verificar se o arquivo é um arquivo CSV e contém "summary" no nome
            if arquivo.endswith('.csv') and 'summary' in arquivo.lower():
                caminho_arquivo = os.path.join(caminho_diretorio, arquivo)
                
                #lendo arquivo
                resultado_ptrails = pd.read_csv(caminho_arquivo)
                
                #lendo dados somente TMT-A e B
                tmt_a = resultado_ptrails[resultado_ptrails['type'] == 1]
                tmt_b = resultado_ptrails[resultado_ptrails['type'] == 2]
                
                #selecionando tentativas corretas
                tmt_a_teste = tmt_a[tmt_a['targs'].isin([26])]
                tmt_b_teste = tmt_b[tmt_b['targs'].isin([26])]
                
                #escrevendo resultados
                linha_ptrails['Numero'] = pd.to_numeric(diretorio)
                linha_ptrails['Media tempo total A'] = tmt_a_teste['totaltime'].mean()
                linha_ptrails['Media tempo total B'] = tmt_b_teste['totaltime'].mean()
                linha_ptrails['Flexibilidade cognitiva media (B-A)'] = linha_ptrails['Media tempo total B'] - linha_ptrails['Media tempo total A']
                linha_ptrails['Melhor tempo total A'] = tmt_a_teste['totaltime'].min()
                linha_ptrails['Melhor tempo total B'] = tmt_b_teste['totaltime'].min()
                linha_ptrails['Flexibilidade cognitiva (B-A)'] = linha_ptrails['Melhor tempo total B'] - linha_ptrails['Melhor tempo total A']
                linha_ptrails['Acuracia media A'] = tmt_a_teste['acc'].mean()
                linha_ptrails['Acuracia media B'] = tmt_b_teste['acc'].mean()
                
                resultados_ptrails = pd.concat([resultados_ptrails, linha_ptrails])
                
    return resultados_ptrails
    
def save_csv(path_to_save, corsi_data, gonogo_data, mot_data, ptrails_data):
    path_corsi = os.path.join(path_to_save, 'corsi_data.csv')
    path_gonogo = os.path.join(path_to_save, 'gonogo_data.csv')
    path_mot = os.path.join(path_to_save, 'mot_data.csv')
    path_ptrails = os.path.join(path_to_save, 'ptrails_data.csv')
    
    corsi_data.to_csv(path_corsi)
    gonogo_data.to_csv(path_gonogo)
    mot_data.to_csv(path_mot)
    ptrails_data.to_csv(path_ptrails)

if __name__ == "__main__":
    
    #Mude o caminho para o de seu computador
    path_to_data = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\data'
    
    # Loop através dos itens no diretório, cada um representando uma coleta de dados cognitivos
    for time in os.listdir(path_to_data):
        # Verificar se o item é a pasta 'results'
        if time == 'results':
            continue  # Ignorar a pasta 'results' e continuar com o próximo item
            
        path_to_corsi = (os.path.join(path_to_data, time, 'corsi'))
        
        path_to_gonogo = (os.path.join(path_to_data, time, 'gonogo'))
        
        path_to_ptrails = (os.path.join(path_to_data, time, 'ptrails'))
        
        path_to_mot = (os.path.join(path_to_data, time, 'mot', 'results'))
        
        corsi_data = read_corsi(path_to_corsi)
        
        gonogo_data = read_gonogo(path_to_gonogo)
        
        mot_data = read_mot(path_to_mot)
        
        ptrails_data = read_ptrails(path_to_ptrails)
        
        #salvar arquivo csv
        path_to_save = os.path.join(path_to_data,'results', time)
        
        # Verificar se o diretório de salvamento não existe e criar se necessário
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        save_csv(path_to_save, corsi_data, gonogo_data, mot_data, ptrails_data)
    
    