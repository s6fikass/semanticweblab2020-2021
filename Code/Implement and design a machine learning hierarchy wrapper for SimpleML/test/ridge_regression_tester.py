from MLalgorithms.Regression._RidgeRegression import RidgeRegression
import numpy as np

n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)
adapterObj = RidgeRegression(alpha=1.0)
adapterObj.fit(X, y)
print(adapterObj.model.coef_)
