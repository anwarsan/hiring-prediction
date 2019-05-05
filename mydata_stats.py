import pandas as pd
import scipy.stats as stats
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind


def chi2_test(df, colx, coly, alpha):
    X = df[colx].astype(str)
    Y = df[coly].astype(str)
    dfObserved = pd.crosstab(Y,X)
    chi2, p, dof, expected = stats.chi2_contingency(dfObserved.values)
    if p < alpha:
        print("%s is important for %s prediction." % (colx, coly))
    else:
        print("%s is NOT important for %s prediction." % (colx, coly))
        
        
def anova_hair(ord_var, df):
    for var in ord_var:   
        print(var)
        result = stats.f_oneway(df[var][df['hair']=='brown'],
                                df[var][df['hair']=='blond'],
                                df[var][df['hair']=='chestnut'],
                                df[var][df['hair']=='red']) 
        print(result)
        

def anova_hiring(ord_var, df):
    for var in ord_var:    
        print(var)
        result = stats.f_oneway(df[var][df['hiring_bis'] == 'yes'], 
                                df[var][df['hiring_bis'] == 'no'])
        print(result)
