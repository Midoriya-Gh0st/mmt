"""
More than 2 views
===========================

This will compare MCCA, GCCA, TCCA for linear models with more than 2 views
"""
import numpy as np

from cca_zoo.data import generate_covariance_data
from cca_zoo.models import MCCA, GCCA, TCCA, KCCA, KGCCA, KTCCA, SCCA_PMD

"""
Data
-----
"""
# %%
np.random.seed(42)
n = 10
p = 3
q = 3
r = 3
latent_dims = 1
# cv = 3

(X, Y, Z), (tx, ty, tz) = generate_covariance_data(
    n, view_features=[p, q, r], latent_dims=latent_dims, correlation=[0.9]
)

# print(f"[XYZ]:", X.shape)
# print(f"[X_True]: {tx}")

# %%
# Eigendecomposition-Based Methods
# ---------------------------------

# %%
# Linear
# ^^^^^^^^

# %%
# X = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
# print(f"[XYZ]:", X.shape)
# Y = np.array([[4, 5, 6], [4, 5, 6], [4, 5, 6], [4, 5, 6]])
# Z = np.array([[7, 8, 9], [7, 8, 9], [7, 8, 9], [7, 8, 9]])
# mcca = MCCA(latent_dims=latent_dims).fit((X, Y, Z)).score((X, Y, Z))
mcca = MCCA(latent_dims=latent_dims).fit((X, Y, Z)).pairwise_correlations((X, Y, Z))

print("the type:", type(mcca), mcca.shape)
print(mcca)

# # %%
# gcca = GCCA(latent_dims=latent_dims).fit((X, Y, X)).score((X, Y, Z))
#
# # %%
# # Kernel
# # ^^^^^^^^
#
# # %%
# kcca = KCCA(latent_dims=latent_dims).fit((X, Y, X)).score((X, Y, Z))
#
# # %%
# kgcca = KGCCA(latent_dims=latent_dims).fit((X, Y, X)).score((X, Y, Z))
#
# # %%
# # Iterative Methods
# # ^^^^^^^^^^^^^^^^^^
#
# # Most of the _iterative methods can also use multiple views e.g.
#
# pmd = SCCA_PMD(latent_dims=latent_dims, c=1).fit((X, Y, X)).score((X, Y, Z))
#
#
# # %%
# # Higher Order Correlations
# # -------------------------
#
# # %%
# # Tensor CCA finds higher order correlations so scores are not comparable (but TCCA is equivalent for 2 views)
# tcca = TCCA(latent_dims=latent_dims).fit((X, Y, X)).score((X, Y, Z))
#
# # %%
# ktcca = KTCCA(latent_dims=latent_dims).fit((X, Y, X)).score((X, Y, Z))