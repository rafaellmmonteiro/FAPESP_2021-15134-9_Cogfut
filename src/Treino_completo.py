import GridSearch
import Validacao_cruzada
import os

if __name__ == "__main__":
            
    path_dir_combinacoes = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\data\\func_cog_misturado'
    
    for root, dirs, files in os.walk(path_dir_combinacoes):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            path_data = os.path.join(path_dir_combinacoes, dir_name)
            path_param = os.path.join('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\best_param\\func_cog_misturado_stratkfold4', dir_name)
            path_save_grid = os.path.join('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\best_param\\func_cog_misturado_stratkfold4', dir_name)
            path_save_valid = os.path.join('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\resultados_ml\\func_cog_misturado_stratkfold4', dir_name)
    
            GridSearch.run_GridSearch(n_clusters_Desempenho_campo = 2, n_splits_kfold = 4,
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
                    path_data = path_data,
                    path_save = path_save_grid)

            Validacao_cruzada.run_validacao_cruzada(path_data = path_data,
                                  path_param = path_param,
                                  path_save = path_save_valid,
                                  n_clusters_Desempenho_campo = 2, n_splits_kfold = 4,
                                  oversample = False, normal = True, metade = False)
            

    path_dir_combinacoes = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\data\\func_cog_misturado'
    
    for root, dirs, files in os.walk(path_dir_combinacoes):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            path_data = os.path.join(path_dir_combinacoes, dir_name)
            path_param = os.path.join('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\best_param\\func_cog_misturado_stratkfold5', dir_name)
            path_save_grid = os.path.join('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\best_param\\func_cog_misturado_stratkfold5', dir_name)
            path_save_valid = os.path.join('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\resultados_ml\\func_cog_misturado_stratkfold5', dir_name)
    
            GridSearch.run_GridSearch(n_clusters_Desempenho_campo = 2, n_splits_kfold = 5,
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
                    path_data = path_data,
                    path_save = path_save_grid)

            Validacao_cruzada.run_validacao_cruzada(path_data = path_data,
                                  path_param = path_param,
                                  path_save = path_save_valid,
                                  n_clusters_Desempenho_campo = 2, n_splits_kfold = 5,
                                  oversample = False, normal = True, metade = False)
        
    path_dir_combinacoes = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\data\\func_cog_misturado_oversample'
    
    for root, dirs, files in os.walk(path_dir_combinacoes):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            path_data = os.path.join(path_dir_combinacoes, dir_name)
            path_param = os.path.join('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\best_param\\func_cog_misturado_oversample_stratkfold5', dir_name)
            path_save_grid = os.path.join('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\best_param\\func_cog_misturado_oversample_stratkfold5', dir_name)
            path_save_valid = os.path.join('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\resultados_ml\\func_cog_misturado_oversample_stratkfold5', dir_name)
    
            GridSearch.run_GridSearch(n_clusters_Desempenho_campo = 2, n_splits_kfold = 5,
                    naive = True,
                    randomforest = True,
                    knn = True,
                    Logistic_Regression = True,
                    SVM = True,
                    MLP = True,
                    XGboost = True,
                    oversample = True,
                    normal = False,
                    metade = False,
                    path_data = path_data,
                    path_save = path_save_grid)
            
            Validacao_cruzada.run_validacao_cruzada(path_data = path_data,
                                  path_param = path_param,
                                  path_save = path_save_valid,
                                  n_clusters_Desempenho_campo = 2, n_splits_kfold = 5,
                                  oversample = True, normal = False, metade = False)
            
    path_dir_combinacoes = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\data\\func_cog_misturado_oversample'
    
    for root, dirs, files in os.walk(path_dir_combinacoes):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            path_data = os.path.join(path_dir_combinacoes, dir_name)
            path_param = os.path.join('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\best_param\\func_cog_misturado_oversample_stratkfold4', dir_name)
            path_save_grid = os.path.join('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\best_param\\func_cog_misturado_oversample_stratkfold4', dir_name)
            path_save_valid = os.path.join('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\resultados_ml\\func_cog_misturado_oversample_stratkfold4', dir_name)
    
            GridSearch.run_GridSearch(n_clusters_Desempenho_campo = 2, n_splits_kfold = 4,
                    naive = True,
                    randomforest = True,
                    knn = True,
                    Logistic_Regression = True,
                    SVM = True,
                    MLP = True,
                    XGboost = True,
                    oversample = True,
                    normal = False,
                    metade = False,
                    path_data = path_data,
                    path_save = path_save_grid)
            
            Validacao_cruzada.run_validacao_cruzada(path_data = path_data,
                                  path_param = path_param,
                                  path_save = path_save_valid,
                                  n_clusters_Desempenho_campo = 2, n_splits_kfold = 4,
                                  oversample = True, normal = False, metade = False)
            
    path_dir_combinacoes = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\data\\func_cog_misturado_oversample'
    
    for root, dirs, files in os.walk(path_dir_combinacoes):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            path_data = os.path.join(path_dir_combinacoes, dir_name)
            path_param = os.path.join('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\best_param\\func_cog_misturado_oversample_stratkfold3', dir_name)
            path_save_grid = os.path.join('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\best_param\\func_cog_misturado_oversample_stratkfold3', dir_name)
            path_save_valid = os.path.join('C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\resultados_ml\\func_cog_misturado_oversample_stratkfold3', dir_name)
    
            GridSearch.run_GridSearch(n_clusters_Desempenho_campo = 2, n_splits_kfold = 3,
                    naive = True,
                    randomforest = True,
                    knn = True,
                    Logistic_Regression = True,
                    SVM = True,
                    MLP = True,
                    XGboost = True,
                    oversample = True,
                    normal = False,
                    metade = False,
                    path_data = path_data,
                    path_save = path_save_grid)
            
            Validacao_cruzada.run_validacao_cruzada(path_data = path_data,
                                  path_param = path_param,
                                  path_save = path_save_valid,
                                  n_clusters_Desempenho_campo = 2, n_splits_kfold = 3,
                                  oversample = True, normal = False, metade = False)

