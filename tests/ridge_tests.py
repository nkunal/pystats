import unittest
import pandas as pd
import ridge
import numpy as np



class RidgeTests(unittest.TestCase):

    def cathetar_test(self):
        cx=pd.DataFrame(pd.read_csv("datasets/cathetar.txt", delim_whitespace=True))
        mod1=ridge.Ridge([0.00001, 0.0001, 0.001, 0.01, 0.1, 0.0])
        mod1.fit(cx, "Catheter_Length", ["Height",  "Weight"])
        self.assertTrue(np.isclose(mod1.final_model.ssr, 607.187798))

    def wine_test(self):
        cx=pd.DataFrame(pd.read_csv("datasets/winequality-red.csv", sep=';'))
        mod1=ridge.Ridge([0.00001, 0.0001, 0.001, 0.01, 0.1, 0.0])
        mod1.fit(cx, "quality", ["fixed acidity", "volatile acidity", "citric acid",
                                 "residual sugar", "chlorides", "free sulfur dioxide",
                                 "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"])
        self.assertTrue(np.isclose(mod1.final_model.ssr, 374.924))
