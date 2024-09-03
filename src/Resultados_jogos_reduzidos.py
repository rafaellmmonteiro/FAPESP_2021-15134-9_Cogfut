import pandas as pd
import os

def atualizar_resultados(performance_jogadores, time):

    ############################rodadas jogadas############################
    performance_jogadores.loc[performance_jogadores['Numero'] == time.iloc[0], 'rodadas'] += 1
    performance_jogadores.loc[performance_jogadores['Numero'] == time.iloc[2], 'rodadas'] += 1
    performance_jogadores.loc[performance_jogadores['Numero'] == time.iloc[4], 'rodadas'] += 1
    
    ############################gols feitos############################
    #Jogador 1 (A ou D)
    performance_jogadores.loc[performance_jogadores['Numero'] == time.iloc[0], 'gols_feitos'] += time.iloc[1]
    
    #Jogador 2 (B ou E)
    performance_jogadores.loc[performance_jogadores['Numero'] == time.iloc[2], 'gols_feitos'] += time.iloc[3]
    
    #Jogador 3 (C ou F)
    performance_jogadores.loc[performance_jogadores['Numero'] == time.iloc[4], 'gols_feitos'] += time.iloc[5]
    
    ############################gols sofridos############################
    performance_jogadores.loc[performance_jogadores['Numero'] == time.iloc[0], 'gols_sofridos'] += time.iloc[6]
    performance_jogadores.loc[performance_jogadores['Numero'] == time.iloc[2], 'gols_sofridos'] += time.iloc[6]
    performance_jogadores.loc[performance_jogadores['Numero'] == time.iloc[4], 'gols_sofridos'] += time.iloc[6]
    
    ############################gols feitos por companheiros de equipe############################
    performance_jogadores.loc[performance_jogadores['Numero'] == time.iloc[0], 'gols_companheiros'] += time.iloc[3] + time.iloc[5]
    performance_jogadores.loc[performance_jogadores['Numero'] == time.iloc[2], 'gols_companheiros'] += time.iloc[1] + time.iloc[5]
    performance_jogadores.loc[performance_jogadores['Numero'] == time.iloc[4], 'gols_companheiros'] += time.iloc[1] + time.iloc[3]
    
    ############################saldo de gols############################
    performance_jogadores.loc[performance_jogadores['Numero'] == time.iloc[0], 'saldo_gols'] += time.iloc[7]
    performance_jogadores.loc[performance_jogadores['Numero'] == time.iloc[2], 'saldo_gols'] += time.iloc[7]
    performance_jogadores.loc[performance_jogadores['Numero'] == time.iloc[4], 'saldo_gols'] += time.iloc[7]
    
    #contar quando ganha
    if time.iloc[7] > 0:
        performance_jogadores.loc[performance_jogadores['Numero'] == time.iloc[0], 'vitorias'] += 1
        performance_jogadores.loc[performance_jogadores['Numero'] == time.iloc[2], 'vitorias'] += 1
        performance_jogadores.loc[performance_jogadores['Numero'] == time.iloc[4], 'vitorias'] += 1
        
    #contar quando empata
    if time.iloc[7] == 0:
        performance_jogadores.loc[performance_jogadores['Numero'] == time.iloc[0], 'empates'] += 1
        performance_jogadores.loc[performance_jogadores['Numero'] == time.iloc[2], 'empates'] += 1
        performance_jogadores.loc[performance_jogadores['Numero'] == time.iloc[4], 'empates'] += 1
        
    #contar quando perde
    if time.iloc[7] < 0:
        performance_jogadores.loc[performance_jogadores['Numero'] == time.iloc[0], 'derrotas'] += 1
        performance_jogadores.loc[performance_jogadores['Numero'] == time.iloc[2], 'derrotas'] += 1
        performance_jogadores.loc[performance_jogadores['Numero'] == time.iloc[4], 'derrotas'] += 1
        
    return performance_jogadores

def analisar_rodada(performance_jogadores, data_rodada):
    
    #gol feitos por cada time
    gols_time_1 = data_rodada.iloc[1] + data_rodada.iloc[3] + data_rodada.iloc[5]
    gols_time_2 = data_rodada.iloc[7] + data_rodada.iloc[9] + data_rodada.iloc[11]
    
    #Saldo de gols time
    saldo_gols_time_1 = gols_time_1-gols_time_2
    saldo_gols_time_2 = gols_time_2-gols_time_1
    
    #dataframe saida com dados da rodada
    time_1 =  pd.concat([data_rodada.iloc[0:6], pd.Series([gols_time_2, saldo_gols_time_1])])
    time_2 =  pd.concat([data_rodada.iloc[6:12], pd.Series([gols_time_1, saldo_gols_time_2])])
    
    #extrair resultados totais
    performance_jogadores = atualizar_resultados(performance_jogadores, time_1)
    performance_jogadores = atualizar_resultados(performance_jogadores, time_2)
    
    return performance_jogadores
    
def calcVariaveis( nomes_numeros_jog, resultados, num_jog, campos = 1, rodadas = 20):
    #cria tabela para os resultados
    performance_jogadores = nomes_numeros_jog.copy()
    performance_jogadores[['gols_feitos', 'gols_sofridos', 'gols_companheiros', 'saldo_gols', 'vitorias', 'empates', 'derrotas', 'rodadas']] = 0
    
    #ajustando resultados
    resultados = resultados.iloc[2:, 3:]
    
    #pecorre todos os campos
    for campo in range(0, campos*12, 12):
        
        #Percorre todas as rodadas
        for rodada in range(0, rodadas+1):
            
            #seleciona os dados da rodada
            data_rodada = resultados.iloc[rodada, campo:campo+12]
            
            #extrai as variaveis dos dados da rodada
            performance_jogadores = analisar_rodada(performance_jogadores, data_rodada)
            
    return performance_jogadores
    
def calcMedias( performance_jogadores):
    performance_jogadores['gols_feitos'] = performance_jogadores['gols_feitos'] /  performance_jogadores['rodadas']
    performance_jogadores['gols_sofridos'] = performance_jogadores['gols_sofridos'] /  performance_jogadores['rodadas']
    performance_jogadores['gols_companheiros'] = performance_jogadores['gols_companheiros'] /  performance_jogadores['rodadas']
    performance_jogadores['saldo_gols'] = performance_jogadores['saldo_gols'] /  performance_jogadores['rodadas']
    
    return  performance_jogadores

def media_jogadores(resultados, num_jog = 20):
    
    #selecionando nomes e numeros dos jogadores
    nomes_numeros_jog = resultados[['Numero', 'Nome_do_jogador']].loc[2:num_jog+1]

    #Calcular gols feitos, sofridos, feitos por companheiros e saldo final
    performance_jogadores_bruto = calcVariaveis(nomes_numeros_jog, resultados, num_jog, campos = 4, rodadas = 28)
    
    #performance dos jogadores média por jogo
    performance_jogadores =  performance_jogadores_bruto.copy()
    performance_jogadores = calcMedias( performance_jogadores)
    
    
    return performance_jogadores_bruto, performance_jogadores
    
def save_csv(path_to_save, performance_jogadores, performance_jogadores_bruto):
    path_performance_jogadores = os.path.join(path_to_save, 'Media_performance_jogadores.csv')
    path_performance_jogadores_bruto = os.path.join(path_to_save, 'Total_performance_jogadores.csv')
    
    performance_jogadores.to_csv(path_performance_jogadores)
    performance_jogadores_bruto.to_csv(path_performance_jogadores_bruto)
    
if __name__ == "__main__":
    
    path_to_data = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\data'
    
    # Loop através dos itens no diretório, cada um representando uma coleta de dados cognitivos
    for time in os.listdir(path_to_data):
        # Verificar se o item é a pasta 'results'
        if time == 'results':
            continue  # Ignorar a pasta 'results' e continuar com o próximo item
        
        # Ler o arquivo Excel
        resultados = pd.read_excel(os.path.join(path_to_data, time, 'jogos_reduzidos', 'Resultado_jogos.xlsx'))
        
        #ver médida de cada jogador
        performance_jogadores_bruto, performance_jogadores= media_jogadores(resultados, num_jog = 30)
        
        #salvar arquivo csv
        path_to_save = os.path.join(path_to_data,'results', time)
        
        # Verificar se o diretório de salvamento não existe e criar se necessário
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        #Salvar arquivo
        save_csv(path_to_save,  performance_jogadores, performance_jogadores_bruto)
    
