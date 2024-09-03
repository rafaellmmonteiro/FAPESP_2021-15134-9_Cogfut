from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import csv
import ast

def extrair_params_dict(linha):
    # Divida a linha pelo caractere de vírgula
    elementos = linha.split(',')

    # Extraia a string de parâmetros
    params_str = ','.join(elementos[2:]).strip('""')

    # Use ast.literal_eval() para avaliar a string de parâmetros como um dicionário Python
    params_dict = ast.literal_eval(params_str)
    
    return params_dict
    
def importances_random_forest_ensemble(modelo_params, identificador, X_cog, y_campo, n_estimators=100000):
    
    # Inicializando matrizes para armazenar as importâncias
    feature_importances = np.zeros(X_cog.shape[1])
    feature_importances_all = np.zeros((n_estimators, X_cog.shape[1]))

    # Treinando o modelo n_estimators vezes
    for seed in range(n_estimators):
        modelo = RandomForestClassifier(random_state=seed, **modelo_params)
        modelo.fit(X_cog, y_campo)
        feature_importances += modelo.feature_importances_
        feature_importances_all[seed, :] = modelo.feature_importances_

    # Calculando a média das importâncias
    feature_importances /= n_estimators
    
    # Calculando o desvio padrão das importâncias
    std_importances = np.std(feature_importances_all, axis=0)

    # Criando um DataFrame para armazenar as importâncias e seus desvios padrões
    importances_pd = pd.DataFrame({
        'Importance': feature_importances,
        'STD': std_importances
    })

    novo_indice = ['Memory span', 'Acuracia Go', 'Acuracia nogo', 'Capacidade de rastreamento', 'Flexibilidade cognitiva (B-A)']
    importances_pd.index = novo_indice
    
    # Ordenando as importâncias
    importances_pd = importances_pd.sort_values(by='Importance', ascending=False)
    indices = importances_pd.index.tolist()

    # Plotando o gráfico com barras de erro
    plt.figure(figsize=(10, 6))
    plt.title(f"Random Forest - Importância dos Atributos - {identificador}")
    plt.bar(range(len(importances_pd)), importances_pd['Importance'], yerr=importances_pd['STD'], align="center", alpha=0.7, ecolor='black', capsize=10)
    plt.xticks(range(len(indices)), indices, rotation=90)
    plt.xlabel("Atributos")
    plt.ylabel("Importância")
    plt.tight_layout()
    plt.show()


def run_modelo_final(path_data, path_param, path_save, n_clusters_Desempenho_campo = 2, oversample = False, normal = False,
                     metade = False, n_estimators = 10):
    
    paths = [os.path.join(path_data, 'cogfut_gf.pkl'),
             os.path.join(path_data, 'cogfut_gs.pkl'),
             os.path.join(path_data, 'cogfut_gc.pkl'),
             os.path.join(path_data, 'cogfut_sg.pkl')]
    
    paths_param = [os.path.join(path_param,'gf_parametros.csv'),
                   os.path.join(path_param,'gs_parametros.csv'),
                   os.path.join(path_param,'gc_parametros.csv'),
                   os.path.join(path_param,'sg_parametros.csv')]
    
    for i, (path, paths_params) in enumerate(zip(paths, paths_param)):
        
        # Extrair o identificador do arquivo (gf, gs, gc, sg)
        identificador = path.split('_')[-1].split('.')[0]
        
        #Lendo arquivo com dados separados em treinamento e teste
        with open(path, 'rb') as f:
            X_cog_treinamento, y_campo_treinamento, X_cog_teste, y_campo_teste = pickle.load(f)
        
        #juntando arquivos para ter todos os dados em um variável
        X_cog = np.concatenate((X_cog_treinamento, X_cog_teste), axis = 0)
        y_campo = np.concatenate((y_campo_treinamento, y_campo_teste), axis = 0)
        
        #lendo parâmetros
        best_param = pd.read_csv(paths_params)
        
        if identificador == 'gf':
            parametros = extrair_params_dict(best_param.iloc[1,0])  
            modelo_final_gf = RandomForestClassifier(random_state = 0, **parametros)
            modelo_final_gf.fit(X_cog, y_campo)
            
            #mostrando importancia
            importances_random_forest_ensemble(parametros, identificador, X_cog, y_campo, n_estimators=n_estimators)
          
            #importances_random_forest(modelo_final_gf, identificador, X_cog)
            
            #Salvando o modelo
            nome_arquivo_csv = f"{identificador}_modelo_final.sav"
            caminho_arquivo = os.path.join(path_save, nome_arquivo_csv)
            pickle.dump(modelo_final_gf, open(caminho_arquivo, 'wb'))
            
        if identificador == 'gs':
            parametros = extrair_params_dict(best_param.iloc[1,0])  
            modelo_final_gs = RandomForestClassifier(random_state = 0, **parametros)
            modelo_final_gs.fit(X_cog, y_campo)
            
            #mostrando importancia
            importances_random_forest_ensemble(parametros, identificador, X_cog, y_campo, n_estimators=n_estimators)
            
            #importances_random_forest(modelo_final_gs, identificador, X_cog)
            
            #Salvando o modelo
            nome_arquivo_csv = f"{identificador}_modelo_final.sav"
            caminho_arquivo = os.path.join(path_save, nome_arquivo_csv)
            pickle.dump(modelo_final_gs, open(caminho_arquivo, 'wb'))
        
        if identificador == 'gc':
            parametros = extrair_params_dict(best_param.iloc[4,0]) 
            modelo_final_gc = SVC(random_state = 0, **parametros)
            modelo_final_gc.fit(X_cog, y_campo)
            
            #Salvando o modelo
            nome_arquivo_csv = f"{identificador}_modelo_final.sav"
            caminho_arquivo = os.path.join(path_save, nome_arquivo_csv)
            pickle.dump(modelo_final_gc, open(caminho_arquivo, 'wb'))
            
        if identificador == 'sg':
            parametros = extrair_params_dict(best_param.iloc[5,0]) 
            modelo_final_sg = MLPClassifier(**parametros)
            modelo_final_sg.fit(X_cog, y_campo)
            
            #Salvando o modelo
            nome_arquivo_csv = f"{identificador}_modelo_final.sav"
            caminho_arquivo = os.path.join(path_save, nome_arquivo_csv)
            pickle.dump(modelo_final_sg, open(caminho_arquivo, 'wb'))
        

          
    return 

if __name__ == "__main__":

    run_modelo_final(path_data = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\data\\Oversample',
                     path_param = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\best_param\\Oversample',
                     path_save = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\modelo_final\\Oversample',
                     n_clusters_Desempenho_campo = 2, oversample = True, normal = False,
                     metade = False, n_estimators=100)