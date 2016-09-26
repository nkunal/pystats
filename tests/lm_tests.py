import numpy as np
import scipy
import pandas as pd
import lm

def lm_cathetar_test():
    cx=pd.DataFrame(pd.read_csv("datasets/cathetar.txt", delim_whitespace=True))
    mod1=lm.linear_model()
    mod1.fit(cx, "Catheter_Length", ["Height",  "Weight"])
    mod1.summary()
    assert np.allclose(mod1.coefficients, [ 20.37576446, 0.21074728, 0.1910949 ])
    assert np.allclose(mod1.ssr, 607.188)
    assert np.allclose(mod1.sse, 128.479)
    assert np.allclose(mod1.ssto, 735.667)
