import numpy as np
import scipy
import pandas as pd
import lm
import unittest

class LinearModelTest(unittest.TestCase):

    def lm_cathetar_test(self):
        cx=pd.DataFrame(pd.read_csv("datasets/cathetar.txt", delim_whitespace=True))
        mod1=lm.LinearModel()
        mod1.fit(cx, "Catheter_Length", ["Height",  "Weight"])
        mod1.summary()
        assert np.allclose(mod1.coefficients, [ 20.37576446, 0.21074728, 0.1910949 ])
        assert np.allclose(mod1.ssr, 607.188)
        assert np.allclose(mod1.sse, 128.479)
        assert np.allclose(mod1.ssto, 735.667)


    def lm_wines_test(self):
        cx=pd.DataFrame(pd.read_csv("datasets/winequality-red.csv", sep=';'))
        mod1=lm.LinearModel()
        mod1.fit(cx, "quality", ["fixed acidity", "volatile acidity", "citric acid",
                             "residual sugar", "chlorides", "free sulfur dioxide",
                                 "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"])
        mod1.summary()
        coeefs = [21.965208449451751, 0.0249905527,  -1.08359026,  -0.182563948,
                  0.0163312698,  -1.87422516,   0.00436133331,
                  -0.00326457970,  -17.8811638,  -0.413653144,
                  0.916334413,   0.276197699]
        for i,value in enumerate(coeefs):
            self.assertTrue(np.isclose(value, mod1.coefficients[i]),
                            "lm_wines_test: Failed for i={0}, coeffs={1:>10.6f} mod1.coeffs={2:>10.6f}".format(i, value, mod1.coefficients[i]))
