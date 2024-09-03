import os
import pandas as pd

def selecionar_melhores_balanced_accuracy(path_dir_combinacoes):
    # Variáveis de interesse
    metricas_interesse = ['balanced_accuracy', 'f1', 'f1_class_0', 'f1_class_1',  
                          'precision', 'precision_class_0', 'precision_class_1', 
                          'recall', 'recall_class_0', 'recall_class_1']
    # Lista de arquivos de resultados a serem processados
    arquivos_resultados_medias = {
        'gf': 'Medias_gf.xlsx',
        'gs': 'Medias_gs.xlsx',
        'gc': 'Medias_gc.xlsx',
        'sg': 'Medias_sg.xlsx'
    }
    
    arquivos_resultados_desvios_padrao = {
        'gf': 'Dp_gf.xlsx',
        'gs': 'Dp_gs.xlsx',
        'gc': 'Dp_gc.xlsx',
        'sg': 'Dp_sg.xlsx'
    }
    
    # Dicionário para armazenar os melhores resultados
    melhores_resultados = {}

    # Percorrendo cada tipo de resultado ('gf', 'gs', 'gc', 'sg')
    for chave, arquivo_medias in arquivos_resultados_medias.items():
        path_arquivo_medias = os.path.join(path_dir_combinacoes, arquivo_medias)
        path_arquivo_desvios_padrao = os.path.join(path_dir_combinacoes, arquivos_resultados_desvios_padrao[chave])
        
        if os.path.exists(path_arquivo_medias) and os.path.exists(path_arquivo_desvios_padrao):
            sheets_dict_medias = pd.read_excel(path_arquivo_medias, sheet_name=None)
            for dicionario in sheets_dict_medias:
                for column in sheets_dict_medias[dicionario].iloc[:,1:]:
                    sheets_dict_medias[dicionario][column] = pd.to_numeric(sheets_dict_medias[dicionario][column], errors='coerce')
                    
            sheets_dict_desvios_padrao = pd.read_excel(path_arquivo_desvios_padrao, sheet_name=None)
            for dicionario in sheets_dict_desvios_padrao:
                for column in sheets_dict_desvios_padrao[dicionario].iloc[:,1:]:
                    sheets_dict_desvios_padrao[dicionario][column] = pd.to_numeric(sheets_dict_desvios_padrao[dicionario][column], errors='coerce')
            
            if 'balanced_accuracy' in sheets_dict_medias and 'balanced_accuracy' in sheets_dict_desvios_padrao:
                balanced_accuracy = sheets_dict_medias['balanced_accuracy'].head(31)
                std_data = sheets_dict_desvios_padrao['balanced_accuracy'].head(31)
                
                # Corrigir a coluna "Rede Neural" se necessário
                for column in balanced_accuracy.iloc[:,1:].columns:
                    balanced_accuracy[column] = pd.to_numeric(balanced_accuracy[column], errors='coerce')
                    
                # Garantir que estamos trabalhando apenas com colunas numéricas
                df_balanced_accuracy = balanced_accuracy.select_dtypes(include='number')
                
                # Selecionar as 5 melhores médias de 'balanced_accuracy'
                maior_valor_por_linha = df_balanced_accuracy.max(axis=1)
                coluna_do_maior_valor = df_balanced_accuracy.idxmax(axis=1)
                
                # Ordenar os maiores valores em ordem decrescente e selecionar os 5 maiores
                maiores_valores = maior_valor_por_linha.nlargest(5)
                colunas_maiores_valores = coluna_do_maior_valor.iloc[maiores_valores.index]
                combinacoes_maiores_valores = balanced_accuracy['variaveis'].iloc[maiores_valores.index]
                
                # Adicionar os dados das melhores médias ao resultados
                resultados = []
                for idx in maiores_valores.index:
                    linha_resultado = {}
                    linha_resultado['Combinações'] = combinacoes_maiores_valores[idx]
                    linha_resultado['Algoritimo'] = colunas_maiores_valores[idx]
                    for metrica in metricas_interesse:
                        if metrica in sheets_dict_medias:
                            df_metrica = sheets_dict_medias[metrica].head(31).select_dtypes(include='number')
                            linha_resultado[metrica] = sheets_dict_medias[metrica].loc[idx, colunas_maiores_valores[idx]]
                            linha_resultado[f'{metrica}_std'] = sheets_dict_desvios_padrao[metrica].loc[idx, colunas_maiores_valores[idx]]
                    resultados.append(linha_resultado)
                
                # Criar DataFrame com os melhores resultados e armazenar no dicionário
                melhores_resultados[chave] = pd.DataFrame(resultados)
        # Salvar os melhores resultados em arquivos Excel
    for chave, df_resultados in melhores_resultados.items():
        nome_arquivo = arquivos_resultados_medias[chave].replace('Medias_', 'Melhores_5_')
        path_saida = os.path.join(path_dir_combinacoes, nome_arquivo)
        df_resultados.to_excel(path_saida, index=False)
        
    return melhores_resultados

if __name__ == "__main__":
    path_dir_combinacoes = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\resultados_ml\\func_cog_misturado_stratkfold5'
    melhores_resultados = selecionar_melhores_balanced_accuracy(path_dir_combinacoes)
    print(melhores_resultados)
