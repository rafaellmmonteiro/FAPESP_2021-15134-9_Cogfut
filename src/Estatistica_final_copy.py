# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:28:28 2024

@author: rafae
"""

import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
import itertools

# Função para o teste de Dunn com ajuste Bonferroni
def dunn_test(df, group_col, val_col):
    unique_groups = df[group_col].unique()
    comparisons = list(itertools.combinations(unique_groups, 2))
    p_values = []

    for group1, group2 in comparisons:
        data1 = df[df[group_col] == group1][val_col]
        data2 = df[df[group_col] == group2][val_col]
        stat, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        p_values.append(p)

    p_adjusted = np.array(p_values) * len(comparisons)  # Bonferroni adjustment
    p_adjusted = np.where(p_adjusted > 1, 1, p_adjusted)  # Limit p-values to a maximum of 1

    result_df = pd.DataFrame(comparisons, columns=['Group1', 'Group2'])
    result_df['p-value'] = p_values
    result_df['p-adjusted'] = p_adjusted

    return result_df

# Função para rodar os testes de normalidade, homogeneidade, ANOVA/Kruskal-Wallis e post-hoc
def run_statistical_tests(df):
    results = {}
    normality_results = {}
    homogeneity_results = None
    esfericidade_results = None
    anova_results = None
    posthoc_results = None
    kruskal_results = None
    all_columns_normal = True

    # Teste de normalidade (Shapiro-Wilk) para cada grupo dentro de cada coluna
    for column in df.columns:
        stat, p_value = stats.shapiro(df[column])
        normality_results[column] = {'statistic': stat, 'p_value': p_value}
        if p_value < 0.05:  # Se p < 0.05, rejeitamos a hipótese nula de normalidade
            all_columns_normal = False

    results['normality'] = normality_results

    # Decidir entre ANOVA ou Kruskal-Wallis
    melted_df = df.melt(var_name='group', value_name='value')
    if all_columns_normal:
        # Teste de homogeneidade de variâncias (Levene)
        stat, p_value = stats.levene(*[df[col] for col in df.columns])
        homogeneity_results = {'statistic': stat, 'p_value': p_value}
        results['homogeneity'] = homogeneity_results

        if p_value < 0.05:  # Se p < 0.05, rejeitamos a hipótese nula de homogeneidade
            all_columns_normal = False

        if all_columns_normal:
            # Se todas as colunas são normais e homogêneas, fazemos ANOVA de uma via
            model = ols('value ~ group', data=melted_df).fit()
            anova_results = sm.stats.anova_lm(model, typ=2)
            results['anova'] = anova_results

            # Post-hoc (Tukey HSD)
            posthoc = pairwise_tukeyhsd(melted_df['value'], melted_df['group'])
            posthoc_results = posthoc.summary()
            results['posthoc'] = posthoc_results
        else:
            # Se alguma coluna não é homogênea, usamos Kruskal-Wallis
            kruskal_stat, kruskal_p = stats.kruskal(*[df[col] for col in df.columns])
            kruskal_results = {'H-statistic': kruskal_stat, 'p_value': kruskal_p}
            results['kruskal'] = kruskal_results

            # Post-hoc (Dunn)
            posthoc_results = dunn_test(melted_df, 'group', 'value')
            results['posthoc'] = posthoc_results
    else:
        # Se alguma coluna não é normal, fazemos Kruskal-Wallis
        kruskal_stat, kruskal_p = stats.kruskal(*[df[col] for col in df.columns])
        kruskal_results = {'H-statistic': kruskal_stat, 'p_value': kruskal_p}
        results['kruskal'] = kruskal_results

        # Post-hoc (Dunn)
        posthoc_results = dunn_test(melted_df, 'group', 'value')
        results['posthoc'] = posthoc_results

    return results

# Carregar o arquivo e processar cada aba
file_path = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\final_analysis\\func_cog_misturado_stratkfold5\\gc_resultados_3.xlsx'
xls = pd.ExcelFile(file_path)
all_sheets_results = {}

for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)
    df = df.dropna()  # Remove linhas com valores nulos
    all_sheets_results[sheet_name] = run_statistical_tests(df)

# Salvar os resultados em um novo arquivo .xlsx
with pd.ExcelWriter(file_path.replace('.xlsx', '_estatistica.xlsx')) as writer:
    for sheet_name, results in all_sheets_results.items():
        normality_df = pd.DataFrame(results['normality']).transpose()
        normality_df.to_excel(writer, sheet_name=f'{sheet_name}_normality')

        if 'homogeneity' in results:
            homogeneity_df = pd.DataFrame([results['homogeneity']])
            homogeneity_df.to_excel(writer, sheet_name=f'{sheet_name}_homogeneity')

        if 'anova' in results:
            anova_df = pd.DataFrame(results['anova'])
            anova_df.to_excel(writer, sheet_name=f'{sheet_name}_anova')
        elif 'kruskal' in results:
            kruskal_df = pd.DataFrame([results['kruskal']])
            kruskal_df.to_excel(writer, sheet_name=f'{sheet_name}_kruskal')

        posthoc_df = pd.DataFrame(data=results['posthoc'])
        posthoc_df.to_excel(writer, sheet_name=f'{sheet_name}_posthoc')
