from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt
import Pre_processing_classification
import numpy as np
import pandas as pd


def run_naive(n_clusters_Desempenho_campo = 2, n_clusters_cognitivo = 7, clusters_cog = False, oversample = False):
    
    Pre_processing_classification.run(n_clusters_Desempenho_campo, n_clusters_cognitivo, 
                                      cluster = clusters_cog, oversample= oversample)
    
    paths = ['C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_gf.pkl',
             'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_gs.pkl',
             'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_gc.pkl',
             'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_sg.pkl']
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Naive Bayes')
    for i, path in enumerate(paths):
        with open(path, 'rb') as f:
            X_cog_treinamento, y_campo_treinamento, X_cog_teste, y_campo_teste = pickle.load(f)
    
        naive_cogfut_data = GaussianNB()
        naive_cogfut_data.fit(X_cog_treinamento, y_campo_treinamento)
    
        previsoes_cogfut = naive_cogfut_data.predict(X_cog_teste)
         
        print(accuracy_score(y_campo_teste, previsoes_cogfut))
        print(confusion_matrix(y_campo_teste, previsoes_cogfut))
    
        cm = ConfusionMatrix(naive_cogfut_data, ax=axs[i // 2, i % 2])  # Subplot correspondente
        cm.fit(X_cog_treinamento, y_campo_treinamento)
        cm.score(X_cog_teste, y_campo_teste)
        axs[i // 2, i % 2].set_title(f"Confusion Matrix - {path.split('_')[1]} - Accuracy: {accuracy_score(y_campo_teste, previsoes_cogfut)}")

    
        print(classification_report(y_campo_teste, previsoes_cogfut))

        plt.tight_layout()  # Ajusta o layout para evitar sobreposição
        plt.show()
        
def run_randomforest(n_clusters_Desempenho_campo = 2, n_clusters_cognitivo = 7, 
                     n_arvores = 10, vizualizar_importancias = False,
                     clusters_cog = False, oversample = False):
    Pre_processing_classification.run(n_clusters_Desempenho_campo, n_clusters_cognitivo, 
                                      cluster = clusters_cog, oversample= oversample)
    
    paths = ['C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_gf.pkl',
             'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_gs.pkl',
             'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_gc.pkl',
             'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_sg.pkl']
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    fig2, axs2 = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Random forest')
    for i, path in enumerate(paths):
        with open(path, 'rb') as f:
            X_cog_treinamento, y_campo_treinamento, X_cog_teste, y_campo_teste = pickle.load(f)
    
        rf_cogfut_data = RandomForestClassifier(n_estimators=n_arvores, criterion='entropy', random_state = 0)
        rf_cogfut_data.fit(X_cog_treinamento, y_campo_treinamento)
    
        previsoes_cogfut = rf_cogfut_data.predict(X_cog_teste)
         
        print(accuracy_score(y_campo_teste, previsoes_cogfut))
        print(confusion_matrix(y_campo_teste, previsoes_cogfut))
    
        cm = ConfusionMatrix(rf_cogfut_data, ax=axs[i // 2, i % 2])  # Subplot correspondente
        cm.fit(X_cog_treinamento, y_campo_treinamento)
        cm.score(X_cog_teste, y_campo_teste)
        axs[i // 2, i % 2].set_title(f"Confusion Matrix - {path.split('_')[1]} - Accuracy: {accuracy_score(y_campo_teste, previsoes_cogfut)}")

    
        print(classification_report(y_campo_teste, previsoes_cogfut))

        plt.tight_layout()  # Ajusta o layout para evitar sobreposição
        plt.show()
        
        if vizualizar_importancias == True:
                
            # Obtendo a importância dos atributos
            importances = rf_cogfut_data.feature_importances_
            
            
            importances_pd = pd.DataFrame(importances)
            

            novo_indice = ['Último item com tentativa completa','Total score','Total tentativas correctas',
                           'Memory span','Numero,Total corretos','Total erros', 'Acuracia Media',
                           'Erros media','Acuracia Go','Acuracia nogo','Tempo resposta Go',
                           'Tempo resposta nogo','Media de acertos','Capacidade de rastreamento',
                           'Melhor tempo total A', 'Melhor tempo total B','Flexibilidade cognitiva (B-A)',
                           'Acuracia media A', 'Acuracia media B', 'Media tempo total A', 
                           'Media tempo total B','Flexibilidade cognitiva media (B-A)']
            
            importances_pd = importances_pd.set_axis(novo_indice, axis=0)
            
            # Ordenando os índices dos atributos em ordem decrescente de importância
            indices = importances_pd.index.tolist()
            
            # Visualizando a importância dos 10 atributos mais importantes
            axs2[i // 2, i % 2].set_title(f"Importância dos Atributos - {path.split('_')[1]}")
            axs2[i // 2, i % 2].bar(range(len(importances_pd)), importances_pd[0], align="center")
            axs2[i // 2, i % 2].set_xticks(range(X_cog_treinamento.shape[1]))
            axs2[i // 2, i % 2].set_xticklabels(indices, rotation=90)  # Rotaciona os rótulos para melhor visualização

def run_KNN(n_clusters_Desempenho_campo = 2, n_clusters_cognitivo = 7, clusters_cog = False, oversample = False):
    
    Pre_processing_classification.run(n_clusters_Desempenho_campo, n_clusters_cognitivo, 
                                      cluster = clusters_cog, oversample= oversample)
    
    paths = ['C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_gf.pkl',
             'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_gs.pkl',
             'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_gc.pkl',
             'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_sg.pkl']
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('KNN (K-Nearest Neighbors)')
    for i, path in enumerate(paths):
        with open(path, 'rb') as f:
            X_cog_treinamento, y_campo_treinamento, X_cog_teste, y_campo_teste = pickle.load(f)
    
        KNN_cogfut_data = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)
        KNN_cogfut_data.fit(X_cog_treinamento, y_campo_treinamento)
    
        previsoes_cogfut = KNN_cogfut_data.predict(X_cog_teste)
         
        print(accuracy_score(y_campo_teste, previsoes_cogfut))
        print(confusion_matrix(y_campo_teste, previsoes_cogfut))
    
        cm = ConfusionMatrix(KNN_cogfut_data, ax=axs[i // 2, i % 2])  # Subplot correspondente
        cm.fit(X_cog_treinamento, y_campo_treinamento)
        cm.score(X_cog_teste, y_campo_teste)
        axs[i // 2, i % 2].set_title(f"Confusion Matrix - {path.split('_')[1]} - Accuracy: {accuracy_score(y_campo_teste, previsoes_cogfut)}")

    
        print(classification_report(y_campo_teste, previsoes_cogfut))

        plt.tight_layout()  # Ajusta o layout para evitar sobreposição
        plt.show()
        
def run_Logistic_Regression(n_clusters_Desempenho_campo = 2, n_clusters_cognitivo = 7, clusters_cog = False, oversample = False):
    
    Pre_processing_classification.run(n_clusters_Desempenho_campo, n_clusters_cognitivo, 
                                      cluster = clusters_cog, oversample= oversample)
    
    paths = ['C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_gf.pkl',
             'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_gs.pkl',
             'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_gc.pkl',
             'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_sg.pkl']
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Logistic Regression')
    for i, path in enumerate(paths):
        with open(path, 'rb') as f:
            X_cog_treinamento, y_campo_treinamento, X_cog_teste, y_campo_teste = pickle.load(f)
    
        Log_reg_cogfut_data = LogisticRegression(random_state = 1)
        Log_reg_cogfut_data.fit(X_cog_treinamento, y_campo_treinamento)
    
        previsoes_cogfut = Log_reg_cogfut_data.predict(X_cog_teste)
         
        print(accuracy_score(y_campo_teste, previsoes_cogfut))
        print(confusion_matrix(y_campo_teste, previsoes_cogfut))
    
        cm = ConfusionMatrix(Log_reg_cogfut_data, ax=axs[i // 2, i % 2])  # Subplot correspondente
        cm.fit(X_cog_treinamento, y_campo_treinamento)
        cm.score(X_cog_teste, y_campo_teste)
        axs[i // 2, i % 2].set_title(f"Confusion Matrix - {path.split('_')[1]} - Accuracy: {accuracy_score(y_campo_teste, previsoes_cogfut)}")

    
        print(classification_report(y_campo_teste, previsoes_cogfut))

        plt.tight_layout()  # Ajusta o layout para evitar sobreposição
        plt.show()
        
def run_SVM(n_clusters_Desempenho_campo = 2, n_clusters_cognitivo = 7, clusters_cog = False, oversample = False):
    
    Pre_processing_classification.run(n_clusters_Desempenho_campo, n_clusters_cognitivo, 
                                      cluster = clusters_cog, oversample= oversample)
    
    paths = ['C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_gf.pkl',
             'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_gs.pkl',
             'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_gc.pkl',
             'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_sg.pkl']
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Support Vector Machine')
    for i, path in enumerate(paths):
        with open(path, 'rb') as f:
            X_cog_treinamento, y_campo_treinamento, X_cog_teste, y_campo_teste = pickle.load(f)
    
        #Testar Kernel ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
        SVM_cogfut_data = SVC(kernel='sigmoid', random_state=1, C = 2.0)
        SVM_cogfut_data.fit(X_cog_treinamento, y_campo_treinamento)
    
        previsoes_cogfut = SVM_cogfut_data.predict(X_cog_teste)
         
        print(accuracy_score(y_campo_teste, previsoes_cogfut))
        print(confusion_matrix(y_campo_teste, previsoes_cogfut))
    
        cm = ConfusionMatrix(SVM_cogfut_data, ax=axs[i // 2, i % 2])  # Subplot correspondente
        cm.fit(X_cog_treinamento, y_campo_treinamento)
        cm.score(X_cog_teste, y_campo_teste)
        axs[i // 2, i % 2].set_title(f"Confusion Matrix - {path.split('_')[1]} - Accuracy: {accuracy_score(y_campo_teste, previsoes_cogfut)}")

    
        print(classification_report(y_campo_teste, previsoes_cogfut))

        plt.tight_layout()  # Ajusta o layout para evitar sobreposição
        plt.show()
        
def run_MLP(n_clusters_Desempenho_campo = 2, n_clusters_cognitivo = 7, clusters_cog = False, oversample= False):
    
    Pre_processing_classification.run(n_clusters_Desempenho_campo, n_clusters_cognitivo, 
                                      cluster = clusters_cog, oversample= oversample)
    
    paths = ['C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_gf.pkl',
             'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_gs.pkl',
             'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_gc.pkl',
             'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_sg.pkl']
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Multilayer Perceptron')
    for i, path in enumerate(paths):
        with open(path, 'rb') as f:
            X_cog_treinamento, y_campo_treinamento, X_cog_teste, y_campo_teste = pickle.load(f)
    
        MLP_cogfut_data = MLPClassifier(max_iter=1000, verbose=True, tol=0.000100,
                                   solver = 'sgd', activation = 'relu',
                                   hidden_layer_sizes = (50,50), random_state=0)
        
        MLP_cogfut_data.fit(X_cog_treinamento, y_campo_treinamento)
    
        previsoes_cogfut = MLP_cogfut_data.predict(X_cog_teste)
         
        print(accuracy_score(y_campo_teste, previsoes_cogfut))
        print(confusion_matrix(y_campo_teste, previsoes_cogfut))
    
        cm = ConfusionMatrix(MLP_cogfut_data, ax=axs[i // 2, i % 2])  # Subplot correspondente
        cm.fit(X_cog_treinamento, y_campo_treinamento)
        cm.score(X_cog_teste, y_campo_teste)
        axs[i // 2, i % 2].set_title(f"Confusion Matrix - {path.split('_')[1]} - Accuracy: {accuracy_score(y_campo_teste, previsoes_cogfut)}")

    
        print(classification_report(y_campo_teste, previsoes_cogfut))

        plt.tight_layout()  # Ajusta o layout para evitar sobreposição
        plt.show()
        
def run_XGboost(n_clusters_Desempenho_campo = 2, n_clusters_cognitivo = 7, clusters_cog = False, oversample = False):
    
    Pre_processing_classification.run(n_clusters_Desempenho_campo, n_clusters_cognitivo, 
                                      cluster = clusters_cog, oversample= oversample)
    
    paths = ['C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_gf.pkl',
             'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_gs.pkl',
             'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_gc.pkl',
             'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\cogfut_sg.pkl']
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Extreme Gradient Boosting')
    for i, path in enumerate(paths):
        with open(path, 'rb') as f:
            X_cog_treinamento, y_campo_treinamento, X_cog_teste, y_campo_teste = pickle.load(f)
    
        XGB_cogfut_data = XGBClassifier(n_estimators = 10,
                                        max_depth = 10, 
                                        learning_rate = 0.01,
                                        subsample = 1,
                                        min_child_weight=1,
                                        gamma=0,
                                        colsample_bytree=1,
                                        colsample_bylevel=1,
                                        colsample_bynode=1,
                                        reg_alpha=0,
                                        reg_lambda=1,
                                        scale_pos_weight=1,
                                        max_delta_step=0,
                                        random_state=0)
        
        XGB_cogfut_data.fit(X_cog_treinamento, y_campo_treinamento)
    
        previsoes_cogfut = XGB_cogfut_data.predict(X_cog_teste)
         
        print(accuracy_score(y_campo_teste, previsoes_cogfut))
        print(confusion_matrix(y_campo_teste, previsoes_cogfut))
    
        cm = ConfusionMatrix(XGB_cogfut_data, ax=axs[i // 2, i % 2])  # Subplot correspondente
        cm.fit(X_cog_treinamento, y_campo_treinamento)
        cm.score(X_cog_teste, y_campo_teste)
        axs[i // 2, i % 2].set_title(f"Confusion Matrix - {path.split('_')[1]} - Accuracy: {accuracy_score(y_campo_teste, previsoes_cogfut)}")

    
        print(classification_report(y_campo_teste, previsoes_cogfut))

        plt.tight_layout()  # Ajusta o layout para evitar sobreposição
        plt.show()

if __name__ == "__main__":

    #run_naive(n_clusters_Desempenho_campo = 2, n_clusters_cognitivo = 2, cluster_cog = True, oversample = False)
    
    #run_randomforest(n_clusters_Desempenho_campo = 2, n_clusters_cognitivo = 7, 
    #                n_arvores = 100, vizualizar_importancias = True,
    #                clusters_cog = False, oversample = False)
    
    #run_KNN(n_clusters_Desempenho_campo = 2, n_clusters_cognitivo = 2, 
    #        clusters_cog = False, oversample = False)
    
    #run_Logistic_Regression(n_clusters_Desempenho_campo = 2, n_clusters_cognitivo = 15, 
    #        clusters_cog = False, oversample = False)
    #run_SVM(n_clusters_Desempenho_campo = 2, n_clusters_cognitivo = 2, 
    #       clusters_cog = False, oversample = False)
    run_MLP(n_clusters_Desempenho_campo = 2, n_clusters_cognitivo = 7, 
            clusters_cog = False, oversample = True)
    #run_XGboost(n_clusters_Desempenho_campo = 2, n_clusters_cognitivo = 2, 
    #       clusters_cog = False, oversample = False)