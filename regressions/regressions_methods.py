import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LassoCV

p_value_threshold = 0.05

def stepwise_model(current_model, X, y):
    try:
        while True:
            # Store current model criteria for comparison
            current_aic = current_model.aic
            current_bic = current_model.bic
            current_adj_r_squared = current_model.rsquared_adj
            
            # Check if any variable has p-value higher than the threshold
            if max(current_model.pvalues) > p_value_threshold:
                # Find the least significant variable and remove it
                worst_feature = X.columns[np.argmax(current_model.pvalues)]
                X_temp = X.drop(columns=[worst_feature])
                
                # Fit a new model without the least significant variable
                model_temp = sm.OLS(y, X_temp).fit()
                # Use robust covariance matrix to handle potential autocorrelation
                model_temp = model_temp.get_robustcov_results(cov_type='HAC', maxlags=0)
                
                # Check if the new model is better
                if model_temp.aic < current_aic and model_temp.bic < current_bic and model_temp.rsquared_adj > current_adj_r_squared:
                    X = X_temp
                    current_model = model_temp
                else:
                    break
            else:
                break
        print(current_model.summary())
    except Exception as e:
        print(e)
    # Print the summary of the final model
    

def lasso_variables(X, y):
    lasso_cv = LassoCV(cv=15, random_state=45)
    lasso_cv.fit(X, y)

    # Display optimal alpha and intercept
    print("Alpha optimal:", lasso_cv.alpha_)
    print("Intercept:", lasso_cv.intercept_)
    
    # Display non-zero coefficients
    coefficients = dict(zip(X.columns, lasso_cv.coef_))
    for k, v in coefficients.items():
        if v != 0:
            print(k, ":", v)
