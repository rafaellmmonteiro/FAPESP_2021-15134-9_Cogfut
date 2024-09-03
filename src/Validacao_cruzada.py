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
import Pre_processing_classification
import numpy as np
import pandas as pd
import os
import csv
import ast
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef
import math

def salvar_resultados_brutos_em_xlsx(resultados, path_save, identificador):
    """
    Salva os resultados brutos em um arquivo XLSX, com cada aba representando uma métrica.

    Args:
    resultados (dict): Dicionário com resultados para cada algoritmo.
    path_save (str): Caminho para salvar o arquivo XLSX.
    identificador (str): Identificador para o nome do arquivo.
    """
    # Verificar se o diretório para salvar o arquivo existe, se não, criar
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    nome_arquivo_xlsx = f"{identificador}_resultados.xlsx"
    caminho_arquivo = os.path.join(path_save, nome_arquivo_xlsx)

    with pd.ExcelWriter(caminho_arquivo, engine='openpyxl') as writer:
        for metrica in resultados['Naive'][0].keys():  # Usar 'Naive' como exemplo para obter as métricas
            df_metrica = pd.DataFrame({
                'Naive': [resultados['Naive'][i][metrica] for i in range(len(resultados['Naive']))],
                'Random forest': [resultados['Random forest'][i][metrica] for i in range(len(resultados['Random forest']))],
                'KNN': [resultados['KNN'][i][metrica] for i in range(len(resultados['KNN']))],
                'Logistica': [resultados['Logistica'][i][metrica] for i in range(len(resultados['Logistica']))],
                'SVM': [resultados['SVM'][i][metrica] for i in range(len(resultados['SVM']))],
                'Rede neural': [resultados['Rede neural'][i][metrica] for i in range(len(resultados['Rede neural']))],
                'Xgboost': [resultados['Xgboost'][i][metrica] for i in range(len(resultados['Xgboost']))],
            })
            df_metrica.to_excel(writer, sheet_name=metrica, index=False)

def extrair_params_dict(linha):
    # Divida a linha pelo caractere de vírgula
    elementos = linha.split(',')

    # Extraia a string de parâmetros
    params_str = ','.join(elementos[2:]).strip('""')

    # Use ast.literal_eval() para avaliar a string de parâmetros como um dicionário Python
    params_dict = ast.literal_eval(params_str)
    
    return params_dict

def extrair_params_dict_correto(linha):

    # Use ast.literal_eval() para avaliar a string de parâmetros como um dicionário Python
    params_dict = ast.literal_eval(linha)
    
    return params_dict

def calcular_medias_metricas(dicts):
    """
    Calcula a média de cada métrica a partir de uma lista de dicionários, ignorando valores NaN.

    Args:
    dicts (list of dict): Lista de dicionários contendo as métricas.

    Returns:
    dict: Dicionário com a média de cada métrica.
    """
    # Dicionário para armazenar médias
    media_dict = {}

    # Verificar se a lista de dicionários está vazia
    if not dicts:
        return media_dict

    # Cálculo da média de cada métrica
    for metrica in dicts[0]:  # Usando dicts[0] para pegar as chaves
        valores = [d[metrica] for d in dicts if not math.isnan(d[metrica])]
        if valores:  # Verificar se a lista de valores não está vazia
            media_dict[metrica] = sum(valores) / len(valores)
        else:
            media_dict[metrica] = float('nan')  # Se todos os valores forem NaN, definir como NaN

    return media_dict

def salvar_resultados_em_xlsx(resultados, path_save):
    """
    Salva os resultados em um arquivo XLSX, com cada aba representando um identificador.

    Args:
    resultados (dict): Dicionário com resultados para cada identificador.
    path_save (str): Caminho para salvar o arquivo XLSX.
    """
    with pd.ExcelWriter(path_save, engine='openpyxl') as writer:
        for identificador, data in resultados.items():
            df = pd.DataFrame([data])
            df.to_excel(writer, sheet_name=identificador, index=False)

def processar_e_salvar_resultados(path_save, resultados_naive, resultados_random_forest,
                                  resultados_knn, resultados_logistica, resultados_svm,
                                  resultados_rede_neural, resultados_xgboost, identificador):
    resultados_dict = {
        'Naive': resultados_naive,
        'Random forest': resultados_random_forest,
        'KNN': resultados_knn,
        'Logistica': resultados_logistica,
        'SVM': resultados_svm,
        'Rede neural': resultados_rede_neural,
        'Xgboost': resultados_xgboost
    }

    media_resultados = {algoritmo: calcular_medias_metricas(resultados) for algoritmo, resultados in resultados_dict.items()}

    nome_arquivo_xlsx = f"{identificador}_medias.xlsx"
    caminho_arquivo = os.path.join(path_save, nome_arquivo_xlsx)
    salvar_resultados_em_xlsx(media_resultados, caminho_arquivo)
    
    # Salvar resultados brutos
    salvar_resultados_brutos_em_xlsx(resultados_dict, path_save, identificador)


def run_validacao_cruzada(path_data, path_param, path_save, n_clusters_Desempenho_campo = 2, oversample = False,
                          normal = False, metade = False, n_splits_kfold = 3):

    paths = [os.path.join(path_data, 'cogfut_gf.pkl'),
             os.path.join(path_data, 'cogfut_gs.pkl'),
             os.path.join(path_data, 'cogfut_gc.pkl'),
             os.path.join(path_data, 'cogfut_sg.pkl')]
    
    paths_param = [os.path.join(path_param,'gf_parametros.csv'),
                   os.path.join(path_param,'gs_parametros.csv'),
                   os.path.join(path_param,'gc_parametros.csv'),
                   os.path.join(path_param,'sg_parametros.csv')]
    
    # Criar o diretório se não existir
    if not os.path.exists(path_save):
        os.makedirs(path_save)
        
    for i, (path, paths_params) in enumerate(zip(paths, paths_param)):
        
        # Extrair o identificador do arquivo (gf, gs, gc, sg)
        identificador = path.split('_')[-1].split('.')[0]
        
        with open(path, 'rb') as f:
            X_cog_treinamento, y_campo_treinamento, X_cog_teste, y_campo_teste = pickle.load(f)
    
        X_cog = np.concatenate((X_cog_treinamento, X_cog_teste), axis = 0)
        y_campo = np.concatenate((y_campo_treinamento, y_campo_teste), axis = 0)
        
        resultados_naive = []
        resultados_random_forest = []
        resultados_knn = []
        resultados_logistica = []
        resultados_svm = []
        resultados_rede_neural = []
        resultados_xgboost = []
        
        best_param = pd.read_csv(paths_params, sep = ',')
        
        #definindo parâmtreos de avaliação para os algoritimos
        scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='macro'),
            'recall': make_scorer(recall_score, average='macro'),
            'f1': make_scorer(f1_score, average='macro'),
            'roc_auc': 'roc_auc',
            'balanced_accuracy': make_scorer(balanced_accuracy_score),
            'mcc': make_scorer(matthews_corrcoef),
            'precision_class_0': make_scorer(precision_score, average=None, labels=[0]),  # Precisão para a classe 0
            'precision_class_1': make_scorer(precision_score, average=None, labels=[1]),  # Precisão para a classe 1
            'recall_class_0': make_scorer(recall_score, average=None, labels=[0]),        # Recall para a classe 0
            'recall_class_1': make_scorer(recall_score, average=None, labels=[1]),        # Recall para a classe 1
            'f1_class_0': make_scorer(f1_score, average=None, labels=[0]),                # F1-score para a classe 0
            'f1_class_1': make_scorer(f1_score, average=None, labels=[1])                 # F1-score para a classe 1
        }
        
        for j in range(30):
          print(j)
          #kfold = KFold(n_splits=4, shuffle=True, random_state=j)
          kfold = StratifiedKFold(n_splits=n_splits_kfold, shuffle=True, random_state=j)
        
          ######################################################################################
          #Naive
          ######################################################################################
          naive = GaussianNB()
          
          if oversample:

                pipeline = Pipeline([
                    ('smote', SMOTE(sampling_strategy='minority', random_state=0)),
                    ('scaler', StandardScaler()),
                    ('clf', naive)
                ])
                scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
                
                if np.any(np.isnan(scores['test_accuracy'])):
                    pipeline = Pipeline([
                        ('smote', SMOTE(sampling_strategy='minority', random_state=0, k_neighbors=3)),
                        ('scaler', StandardScaler()),
                        ('clf', naive)
                    ])
                scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
                 
          else:
                scores = cross_validate(naive, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
                
          resultados_naive.append({metric: np.mean(scores['test_' + metric]) for metric in scoring})
          
          ######################################################################################
          #random forest
          ######################################################################################
          try:
              parametros = extrair_params_dict(best_param.iloc[1,0])
          except:
              parametros = extrair_params_dict_correto(best_param.iloc[1,2])
              
          random_forest = RandomForestClassifier(**parametros)
          
          if oversample:

                pipeline = Pipeline([
                    ('smote', SMOTE(sampling_strategy='minority', random_state=0)),
                    ('scaler', StandardScaler()),
                    ('clf', random_forest)
                ])
                scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
                
                if np.any(np.isnan(scores['test_accuracy'])):
                        pipeline = Pipeline([
                            ('smote', SMOTE(sampling_strategy='minority', random_state=0, k_neighbors=3)),
                            ('scaler', StandardScaler()),
                            ('clf', random_forest)
                        ])
                        scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
                
          else:
                scores = cross_validate(random_forest, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
          
          resultados_random_forest.append({metric: np.mean(scores['test_' + metric]) for metric in scoring})
          ######################################################################################
          #KNN
          ######################################################################################
          try:
              parametros = extrair_params_dict(best_param.iloc[2,0])
          except:
              parametros = extrair_params_dict_correto(best_param.iloc[2,2])
              
          knn = KNeighborsClassifier(**parametros)
          
          if oversample:

                pipeline = Pipeline([
                    ('smote', SMOTE(sampling_strategy='minority', random_state=0)),
                    ('scaler', StandardScaler()),
                    ('clf', knn)
                ])
                scores = scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
                
                if np.any(np.isnan(scores['test_accuracy'])):
                    pipeline = Pipeline([
                        ('smote', SMOTE(sampling_strategy='minority', random_state=0, k_neighbors=3)),
                        ('scaler', StandardScaler()),
                        ('clf', knn)
                    ])
                    scores = scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
                
          else:
                scores = scores = cross_validate(knn, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
                
          resultados_knn.append({metric: np.mean(scores['test_' + metric]) for metric in scoring})
          
          ######################################################################################
          #logistica
          ######################################################################################
          try:
              parametros = extrair_params_dict(best_param.iloc[3,0])
          except:
              parametros = extrair_params_dict_correto(best_param.iloc[3,2])
              
          logistica = LogisticRegression(**parametros)
          
          if oversample:

                pipeline = Pipeline([
                    ('smote', SMOTE(sampling_strategy='minority', random_state=0)),
                    ('scaler', StandardScaler()),
                    ('clf', logistica)
                ])
                scores =  scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
                
                if np.any(np.isnan(scores['test_accuracy'])):
                    pipeline = Pipeline([
                        ('smote', SMOTE(sampling_strategy='minority', random_state=0, k_neighbors=3)),
                        ('scaler', StandardScaler()),
                        ('clf', logistica)
                    ])
                    scores =  scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
                
          else:
                scores =  scores = cross_validate(logistica, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
                
          resultados_logistica.append({metric: np.mean(scores['test_' + metric]) for metric in scoring})
          
          ######################################################################################
          #svm
          ######################################################################################
          try:
              parametros = extrair_params_dict(best_param.iloc[4,0])
          except:
              parametros = extrair_params_dict_correto(best_param.iloc[4,2])
              
          svm = SVC(**parametros)
          
          if oversample:

                pipeline = Pipeline([
                    ('smote', SMOTE(sampling_strategy='minority', random_state=0)),
                    ('scaler', StandardScaler()),
                    ('clf', svm)
                ])
                scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
                
                if np.any(np.isnan(scores['test_accuracy'])):
                    pipeline = Pipeline([
                        ('smote', SMOTE(sampling_strategy='minority', random_state=0, k_neighbors=3)),
                        ('scaler', StandardScaler()),
                        ('clf', svm)
                    ])
                    scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
                
          else:
                scores = cross_validate(svm, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
                
          resultados_svm.append({metric: np.mean(scores['test_' + metric]) for metric in scoring})
          
          ######################################################################################          
          #rede_neural
          ######################################################################################
          try:
              parametros = extrair_params_dict(best_param.iloc[5,0])
          except:
              parametros = extrair_params_dict_correto(best_param.iloc[5,2])
              
          rede_neural = MLPClassifier(**parametros)
          
          if oversample:

                pipeline = Pipeline([
                    ('smote', SMOTE(sampling_strategy='minority', random_state=0)),
                    ('scaler', StandardScaler()),
                    ('clf', rede_neural)
                ])
                scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
                
                if np.any(np.isnan(scores['test_accuracy'])):
                    pipeline = Pipeline([
                        ('smote', SMOTE(sampling_strategy='minority', random_state=0, k_neighbors=3)),
                        ('scaler', StandardScaler()),
                        ('clf', rede_neural)
                    ])
                    scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
                
          else:
                scores = cross_validate(rede_neural, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
                
          resultados_rede_neural.append({metric: np.mean(scores['test_' + metric]) for metric in scoring})
          
          ######################################################################################
          #Xgboost 
          ######################################################################################
          try:
              parametros = extrair_params_dict(best_param.iloc[6,0])
          except:
              parametros = extrair_params_dict_correto(best_param.iloc[6,2])
              
          Xgboost = XGBClassifier(**parametros)
          
          if oversample:

                pipeline = Pipeline([
                    ('smote', SMOTE(sampling_strategy='minority', random_state=0)),
                    ('scaler', StandardScaler()),
                    ('clf', Xgboost)
                ])
                scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
                
                if np.any(np.isnan(scores['test_accuracy'])):
                    pipeline = Pipeline([
                        ('smote', SMOTE(sampling_strategy='minority', random_state=0, k_neighbors=3)),
                        ('scaler', StandardScaler()),
                        ('clf', Xgboost)
                    ])
                    scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
                
          else:
                scores = cross_validate(Xgboost, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
                
          resultados_xgboost.append({metric: np.mean(scores['test_' + metric]) for metric in scoring})
          
        if identificador == 'gf':
            processar_e_salvar_resultados(path_save, resultados_naive, resultados_random_forest, resultados_knn, resultados_logistica, resultados_svm, resultados_rede_neural, resultados_xgboost, 'gf')
                
        if identificador == 'gs':
            processar_e_salvar_resultados(path_save, resultados_naive, resultados_random_forest, resultados_knn, resultados_logistica, resultados_svm, resultados_rede_neural, resultados_xgboost, 'gs')
                
        if identificador == 'gc':
            processar_e_salvar_resultados(path_save, resultados_naive, resultados_random_forest, resultados_knn, resultados_logistica, resultados_svm, resultados_rede_neural, resultados_xgboost, 'gc')
                      
        if identificador == 'sg':
            processar_e_salvar_resultados(path_save, resultados_naive, resultados_random_forest, resultados_knn, resultados_logistica, resultados_svm, resultados_rede_neural, resultados_xgboost, 'sg')
                  

if __name__ == "__main__":

    run_validacao_cruzada(path_data = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\data\\full',
                          path_param = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\best_param\\full',
                          path_save = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\resultados_ml\\full',
                          n_clusters_Desempenho_campo = 2, n_splits_kfold = 3,
                          oversample = False, normal = True, metade = False)
