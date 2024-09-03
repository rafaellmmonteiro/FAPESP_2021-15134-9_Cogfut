from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import csv
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold

def run_GridSearch(path_data, path_save, n_splits_kfold, n_clusters_Desempenho_campo = 2,
                   naive = False,
                   randomforest = False, 
                   knn = False,
                   Logistic_Regression = False,
                   SVM = False,
                   MLP = False,
                   XGboost = False,
                   oversample = False,
                   normal = False,
                   metade = False):
    
    paths = [os.path.join(path_data, 'cogfut_gf.pkl'),
             os.path.join(path_data, 'cogfut_gs.pkl'),
             os.path.join(path_data, 'cogfut_gc.pkl'),
             os.path.join(path_data, 'cogfut_sg.pkl')]
    
    for i, path in enumerate(paths):
        
        # Extrair o identificador do arquivo (gf, gs, gc, sg)
        identificador = path.split('_')[-1].split('.')[0]
        
        with open(path, 'rb') as f:
            X_cog_treinamento, y_campo_treinamento, X_cog_teste, y_campo_teste = pickle.load(f)
    
        X_cog = np.concatenate((X_cog_treinamento, X_cog_teste), axis = 0)
        y_campo = np.concatenate((y_campo_treinamento, y_campo_teste), axis = 0)
        
        if naive == True:
            
            ml_algorithim = GaussianNB()
            parametros = {}
            
            grid_search(ml_algorithim, parametros, X_cog, y_campo, identificador, path_save, oversample, n_splits_kfold)
        
        if randomforest == True:
            
            ml_algorithim = RandomForestClassifier()
            parametros = {'criterion': ['gini', 'entropy'],
                  'n_estimators': [10, 40, 100, 150],
                  'min_samples_split': [2, 5, 10],
                  'min_samples_leaf': [1, 5, 10]}
            
            grid_search(ml_algorithim, parametros, X_cog, y_campo, identificador, path_save, oversample, n_splits_kfold)
            
        if knn == True:
            
            ml_algorithim = KNeighborsClassifier()
            parametros = {'n_neighbors': [1, 2, 3, 5, 10, 20],
                'p': [1, 2]}
            
            grid_search(ml_algorithim, parametros, X_cog, y_campo, identificador, path_save, oversample, n_splits_kfold)
            
        if Logistic_Regression == True:
            
            ml_algorithim = LogisticRegression()
            parametros = {'tol': [0.01, 0.001, 0.0001],
              'C': [1.0, 1.5, 2.0],
              'solver': ['lbfgs', 'sag', 'saga']}
            
            grid_search(ml_algorithim, parametros, X_cog, y_campo, identificador, path_save, oversample, n_splits_kfold)
        
        if SVM == True:
            
            ml_algorithim = SVC()
            parametros = {'tol': [0.001, 0.0001, 0.00001],
              'C': [1.0, 1.5, 2.0],
              'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}
            
            grid_search(ml_algorithim, parametros, X_cog, y_campo, identificador, path_save, oversample, n_splits_kfold)
        
        if MLP == True:
            
            ml_algorithim = MLPClassifier()
            
            parametros = {'max_iter': [100, 500, 1000, 1500],
                          'activation': ['relu', 'logistic', 'tanh'],
                          'solver': ['adam', 'sgd'],
                          'hidden_layer_sizes': [(5,5),(10,10), (25,25), (50,50)],
                          'random_state': [0]}

            grid_search(ml_algorithim, parametros, X_cog, y_campo, identificador, path_save, oversample, n_splits_kfold)
            print('Grid_completo', identificador)

        if XGboost == True:
            
            ml_algorithim = XGBClassifier()
            
            parametros = {'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'n_estimators': [100, 200],
                'min_child_weight': [1, 5]}
            
            grid_search(ml_algorithim, parametros, X_cog, y_campo, identificador, path_save, oversample, n_splits_kfold)
            print('Grid_completo', identificador)
            
    return()

def grid_search(ml_algorithm, parametros, X_cog, y_campo, identificador, path_save, oversample, n_splits_kfold):
    kfold = StratifiedKFold(n_splits=n_splits_kfold, shuffle=True, random_state=0)
    # Criar o diretório se não existir
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    
    if oversample == True:
        # Defina o pipeline com SMOTE, StandardScaler e o algoritmo de aprendizado de máquina
        pipeline = Pipeline([
            ('smote', SMOTE(sampling_strategy='minority', random_state=0)),
            ('scaler', StandardScaler()),
            ('clf', ml_algorithm)
        ])
        
        parametros = {f'clf__{k}': v for k, v in parametros.items()}
    
        grid_search = GridSearchCV(estimator=pipeline, cv=kfold, param_grid=parametros, scoring='balanced_accuracy', refit='f1_macro')
        
    else:
        grid_search = GridSearchCV(estimator=ml_algorithm, cv=kfold, param_grid=parametros, scoring='balanced_accuracy', refit='f1_macro')
        
    grid_search.fit(X_cog, y_campo)
    melhores_parametros = grid_search.best_params_
    melhor_resultado = grid_search.best_score_
    print(melhores_parametros)
    print(melhor_resultado)
    
    # Remover o prefixo 'clf__' dos parâmetros
    melhores_parametros = {k.replace('clf__', ''): v for k, v in melhores_parametros.items()}
    
    # Salvar os melhores parâmetros e resultado em um arquivo CSV
    nome_arquivo_csv = f"{identificador}_parametros.csv"
    caminho_arquivo = os.path.join(path_save, nome_arquivo_csv)
    
    # Verificar se o arquivo já existe
    if os.path.isfile(caminho_arquivo):
        # Abrir o arquivo CSV em modo de escrita
        with open(caminho_arquivo, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            lines = list(reader)
        
            # Procurar por uma linha com o nome do algoritmo
            linha_existente = None
            for idx, line in enumerate(lines):
                if len(line) > 1 and line[0] == ml_algorithm.__class__.__name__:
                    linha_existente = idx
                    break
        
        # Se uma linha com o nome do algoritmo existir, substitua os dados nessa linha
        if linha_existente is not None:
            lines[linha_existente] = [ml_algorithm.__class__.__name__, melhor_resultado, melhores_parametros]
        else:
            # Caso contrário, adicione uma nova linha com os dados
            lines.append([ml_algorithm.__class__.__name__, melhor_resultado, melhores_parametros])
        
        # Escrever os dados de volta para o arquivo CSV
        with open(caminho_arquivo, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(lines)
    else:
        # Abrir o arquivo CSV em modo de escrita
        with open(caminho_arquivo, 'w', newline='') as csvfile:
            # Criar um escritor CSV
            writer = csv.writer(csvfile)
            # Escrever os cabeçalhos
            writer.writerow(['Nome do Algoritmo', 'Melhor Resultado', 'Melhores Parametros'])
            # Escrever os dados na nova linha
            writer.writerow([ml_algorithm.__class__.__name__, melhor_resultado, melhores_parametros])    
    

if __name__ == "__main__":

    run_GridSearch(n_clusters_Desempenho_campo = 2, n_splits_kfold = 3, 
            naive = True,
            randomforest = True,
            knn = True,
            Logistic_Regression = True,
            SVM = True,
            MLP = True,
            XGboost = True,
            oversample = False,
            normal = True,
            metade = False,
            path_data = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\data\\full',
            path_save = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\best_param\\full')