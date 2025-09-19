import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import copy
sns.set()
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Function to create derived columns##################################################################
def create_derived_columns(df):
    df['LogBiddingPrice'] = np.log1p(df['BiddingPrice'].clip(lower=0).astype(float))
    
    df['LogTotalGFA'] = np.log1p(df['TotalGFA'].clip(lower=0).astype(float))
    
    df['TeamSizePerGFA'] = df['TeamTotal'].astype(float) / df['TotalGFA'].replace({0: np.nan})
    df['TeamSizePerGFA'] = df['TeamSizePerGFA'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    df['TeamSizePerLogGFA'] = df['TeamTotal'].astype(float) / df['LogTotalGFA'].replace({0: np.nan})
    df['TeamSizePerLogGFA'] = df['TeamSizePerGFA'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    return df


# Function to sort predictors by R-squared##################################################################
def sort_predictors_by_r_squared(train_df, target_Y):
    result_dict = {}
    for paraX in train_df.columns:
        if paraX != target_Y and paraX != 'BiddingPrice':
            model = smf.ols(f'{target_Y} ~ {paraX}', data=train_df).fit()
            R2 = model.rsquared
            result_dict[paraX] = R2
    result_dict = dict(sorted(result_dict.items(), key=lambda item: item[1], reverse=True))
    # sorted_predictors = list(result_dict.keys())
    # sorted_r_squareds = list(result_dict.values())  
    return result_dict


# Function to test higher order polynomial regression##################################################################
def polynomaial_degree_selection(train_df, best_predictors, best_r_squareds, target_Y):
    polynomial_info = {}
    for i,(pred,degree1_r2) in enumerate(zip(best_predictors, best_r_squareds)):
        best_degree, best_r_squared = test_higher_order_polynomial(train_df, pred, target_Y,max_degree = 10)
        print (f'Checking Parameter --{pred}-- Ranking --{i+1}')
        
        if degree1_r2 > best_r_squared:
            print (f"Since the R-squared did not improved from {degree1_r2:.4f} to {best_r_squared:.4f}, we will keep the linear term for {pred}.")
            print (f'Linear R-squared: {degree1_r2:.4f}, Higher Polynomial R-squared: {best_r_squared:.4f}')
            polynomial_info[pred] = {'degree': 1, 'R-squared': degree1_r2,'Ranking': i+1}
        else:
            if abs(degree1_r2 - best_r_squared) < 0.1:
                print (f"Since the R-squared improvement is less than 0.1, we will keep the linear term for {pred}.")
                print (f'Linear R-squared: {degree1_r2:.4f}, Higher Polynomial R-squared: {best_r_squared:.4f}')
                polynomial_info[pred] = {'degree': 1, 'R-squared': degree1_r2,'Ranking': i+1}
            else:
                print (f"We will use polynomial degree {best_degree} for {pred}.")
                print (f'Linear R-squared: {degree1_r2:.4f}, Higher Polynomial R-squared: {best_r_squared:.4f}')
                polynomial_info[pred] = {'degree': best_degree, 'R-squared': best_r_squared,'Ranking': i+1}
        print ('')

    # print  (polynomial_info)
    return polynomial_info

def test_higher_order_polynomial(df, predictor, target, max_degree = 5):
    result_dict = {}
    for i in range(max_degree+1):
        if i != 0:
            full_term = ""
            for k in range(i):
                if k > 0:
                    term =  f" + np.power({predictor},{k+1})"
                    full_term += term
            model = smf.ols(f'{target} ~ {predictor}' + full_term, data=df).fit()
            R2 = model.rsquared
            result_dict[i] = R2
        else:
            model = smf.ols(f'{target} ~ {predictor}', data=df).fit()
            R2 = model.rsquared
            result_dict[i] = R2
            
    result_dict = dict(sorted(result_dict.items(), key=lambda item: item[1], reverse=True))
    best_degree = list(result_dict.keys())[0]
    best_r_squared = list(result_dict.values())[0]
    # print (f"The best polynomial degree is {best_degree} with R-squared = {best_r_squared:.4f}")
    
    return best_degree, best_r_squared


#######################################################################################
# Load cleaned data and create derived columns
df = pd.read_pickle("pickle file/Step1_cleaned_df.pkl")
df = create_derived_columns(df)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)
target_Y = 'LogBiddingPrice'

# Check and rank predictors by R-squared
sorted_para_dict = sort_predictors_by_r_squared(train_df, target_Y)
sorted_predictors = list(sorted_para_dict.keys())
sorted_r_squareds = list(sorted_para_dict.values())  

# Check higher order polynomial regression for top all predictors and ranking them
dict_polynomial_info = polynomaial_degree_selection(train_df, sorted_predictors, sorted_r_squareds, target_Y)
print (dict_polynomial_info)