import pandas as pd
import scipy.stats as stats
import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols
    
import matplotlib.pyplot as plt



import numpy as np
from sklearn.linear_model import LinearRegression







# # Loading data
# df = pd.read_csv("https://raw.githubusercontent.com/Opensourcefordatascience/Data-sets/master/difficile.csv")
# df.drop('person', axis= 1, inplace= True)

# # Recoding value from numeric to string
# df['dose'].replace({1: 'placebo', 2: 'low', 3: 'high'}, inplace= True)

# # Gettin summary statistics
# print(rp.summary_cont(df['libido']))

# # group by dose
# print(rp.summary_cont(df['libido'].groupby(df['dose'])))

# # lets do kruskal-wallis? (cause it's non normal)


# # fit into OLS regression result...
# # model_name = ols('outcome_variable ~ group1 + group2 + groupN', data=your_data).fit()
# # model_name = ols('outcome_variable ~ C(group_variable)', data=your_data).fit()

# results = ols('libido ~ C(dose)', data=df).fit()
# print(results.summary())

#aov_table = sm.stats.anova_lm(results, typ=2)
#print(aov_table)

# statistical power (effect power)
#anova_table(aov_table)
#print(results.diagn)

