import pandas as pd
import scipy.stats as stats
import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols
    
import matplotlib.pyplot as plt

def anova_table(aov):
    aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']
    
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])
    
    cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    aov = aov[cols]
    return aov

# Loading data
df = pd.read_csv("https://raw.githubusercontent.com/Opensourcefordatascience/Data-sets/master/difficile.csv")
df.drop('person', axis= 1, inplace= True)

# Recoding value from numeric to string
df['dose'].replace({1: 'placebo', 2: 'low', 3: 'high'}, inplace= True)
    
# Gettin summary statistics
print(rp.summary_cont(df['libido']))

# group by dose
print(rp.summary_cont(df['libido'].groupby(df['dose'])))


# lets do one way anova
anova = stats.f_oneway(df['libido'][df['dose'] == 'high'], 
             df['libido'][df['dose'] == 'low'],
             df['libido'][df['dose'] == 'placebo'])

print(anova)

# fit into OLS regression result...
# model_name = ols('outcome_variable ~ group1 + group2 + groupN', data=your_data).fit()
# model_name = ols('outcome_variable ~ C(group_variable)', data=your_data).fit()

results = ols('libido ~ C(dose)', data=df).fit()
print(results.summary())

#aov_table = sm.stats.anova_lm(results, typ=2)
#print(aov_table)

# statistical power (effect power)
anova_table(aov_table)
#print(results.diagn)

