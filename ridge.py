import numpy as np
import scipy
import pandas as pd

class Ridge:
    def __init__(self, lambda_arr):
        self.lambda_arr = lambda_arr
        self.final_model = None
        self.temp_models = []
    def fit(self, df, response_name, pred_names, method="qr"):
        for value in self.lambda_arr:
            print "Modelling for lambda = %s" %value
            mod1 = RidgeModel(value)
            mod1.fit(df, response_name, pred_names, method)
            self.temp_models.append(mod1)
            if self.final_model is None:
                self.final_model = mod1
            elif self.final_model.ssr > mod1.ssr:
                self.final_model = mod1
    def summary(self):
        for mod1 in self.temp_models:
            print "lambda={0:5} ssr={1:5f}".format(mod1.lambda_val, mod1.ssr)
        print "Final model chosen with lambda=%s" %self.final_model.lambda_val
        self.final_model.summary()



class RidgeModel:
    """
    Class for linear regression model
    """
    def __init__(self, lambda_val):
        self.pred_names = None
        self.coefficients = None
        self.lambda_val = lambda_val
        self.ssr = None
        self.sse = None
        self.ssto = None
        self.coeff_std_err = []
        self.t_values = []
    def fit(self, df, response_name, pred_names, method="qr"):
        """
        Keyword Arguments:
        df -- The predictor matrix, pandas dataframe
        response_name -- The response vector, pandas series /np.ndarray
        pred_names -- Predictor columns to use from df, specify indices
        of columns df. e.g: [0,1,2]

        Fits a multiple linear regression model using
        the foll formula:

        (X^t * X + Lambda * IdentityMatrix) ^ -1 * X^t * response_name
        where:
        XT => Transpose of X matrix
        Inv => Inverse

        Std. Err of Coeeficients Calc:
        coeff_mat = (SSE/n-p-1) * (XT*X)Inv
        std.err.coeffs = sqrt(Diagonal elements of coeff_mat)

        t-value Calculation:
        coeeficient_estimate / std.err.coefficient
        """
        if type(df) != pd.DataFrame:
            print "ERROR: 'df' is not a pandas DataFrame"
            return
        if response_name not in df:
            print "ERROR: response_name not present in df" %response_name
            return
        self.pred_names = pred_names
        y = df[response_name]
        X = df[pred_names]
        X.insert(0, 'betas', 1)
        identity_matrix = np.eye(X.shape[1])
        prod1 = np.linalg.inv(X.T.dot(X) + self.lambda_val * identity_matrix)
        prod2 = X.T.dot(y)
        self.coefficients = prod1.dot(prod2)
        y_hat = X.dot(self.coefficients)
        y_mean = np.mean(y)
        self.ssto = sum(y.apply(lambda values: (y_mean - values) ** 2))
        self.ssr = sum(y_hat.apply(lambda values: (y_mean - values) **2))
        self.sse = sum((y_hat - y).apply(lambda values: values **2))
        coeff_mul = self.sse/ (X.shape[0] - len(self.pred_names) - 1)
        coeff_mat = coeff_mul * prod1
        self.coeff_std_err = np.sqrt(np.diagonal(coeff_mat))
        for i,_ in enumerate(self.coefficients):
            self.t_values.append(self.coefficients[i]/self.coeff_std_err[i])
    def predict(self, df):
        X = df[self.pred_names]
        X.insert(0, 'betas', 1)
        y_hat = X.dot(self.coefficients)
        return y_hat
    def summary(self):
        print "\n**** Ridge Regression Model ****"
        print "Coefficients: "
        print "\t\t{0:20} {1:10} {2:10}".format("Estimate", "Std.Error", "t-value")
        print "{0:20} {1:>10.4f} {2:>10.4f} {3:>10.4f}".format("(Intercept)", self.coefficients[0], self.coeff_std_err[0], self.t_values[0])
        for i, pred in enumerate(self.pred_names):
            print "{0:20} {1:>10.4f} {2:>10.4f} {3:>10.4f}".format(pred, self.coefficients[1+i], self.coeff_std_err[i+1], self.t_values[i+1])
        print "\n"
        print "** ANOVA Table **"
        print "\tSSR:  %0.3f" %self.ssr
        print "\tSSE:  %0.3f" %self.sse
        print "\tSSTO: %0.3f" %self.ssto

if __name__ == "__main__":
    cx=pd.DataFrame(pd.read_csv("datasets/cathetar.txt", delim_whitespace=True))
    mod1=Ridge([0.00001, 0.0001, 0.001, 0.01, 0.1, 0.0])
    mod1.fit(cx, "Catheter_Length", ["Height",  "Weight"])
    mod1.summary()

    cx=pd.DataFrame(pd.read_csv("datasets/winequality-red.csv", sep=';'))
    mod1=Ridge([0.00001, 0.0001, 0.001, 0.01, 0.1, 0.0])
    mod1.fit(cx, "quality", ["fixed acidity", "volatile acidity", "citric acid",
                             "residual sugar", "chlorides", "free sulfur dioxide",
                             "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"])
    mod1.summary()
