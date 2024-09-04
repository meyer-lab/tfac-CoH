"""This creates Figure 8, an examination of tucker decomposition for response data."""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from tensorpack.decomposition import Decomposition
from tensorpack.plot import tucker_reduction
from tensorpack.tucker import tucker_decomp
from tensorly.decomposition import tucker
from itertools import product

from ..flow import get_status_df, make_CoH_Tensor
from ..flow_rec import get_status_rec_df, make_CoH_Tensor_rec
from .common import getSetup, subplotLabel


def makeFigure():
    """ Tucker decomposition figure """
    ax, f = getSetup((12, 8), (2, 3))
    subplotLabel(ax)

    #CoH_Data = make_CoH_Tensor(just_signal=True)
    CoH_Data = make_CoH_Tensor_rec()
    data = CoH_Data.to_numpy()
    mask = np.isfinite(data)

    tres = tucker(np.nan_to_num(data), rank=(8, 6, 4, 6), mask=mask, svd='randomized_svd')
    R2X = 1 - np.nansum((data - tres.to_tensor()) ** 2) / np.nansum(data ** 2)
    # CP 12 comp = 0.8058984095216497

    # Tucker rank explore
    start_rank = [1] * data.ndim
    tucker_rank = [start_rank]
    ttry = tucker(np.nan_to_num(data), rank=start_rank, mask=mask, svd='randomized_svd')
    tucker_R2X = [1 - np.nansum((data - ttry.to_tensor()) ** 2) / np.nansum(data ** 2)]
    for ri in range(100):
        try_ranks, try_R2X = [], []
        for add_i in range(data.ndim):
            new_rank = start_rank.copy()
            if new_rank[add_i] >= data.shape[add_i]:
                continue
            new_rank[add_i] = new_rank[add_i] + 1
            ttry = tucker(np.nan_to_num(data), rank=new_rank, mask=mask, svd='randomized_svd')
            try_ranks.append(new_rank)
            try_R2X.append(1 - np.nansum((data - ttry.to_tensor()) ** 2) / np.nansum(data ** 2))
        if len(try_ranks) <= 0:
            break
        tucker_rank.append(try_ranks[np.argmax(try_R2X)])
        tucker_R2X.append(np.max(try_R2X))
        start_rank = try_ranks[np.argmax(try_R2X)]
    tucker_ranks = pd.DataFrame({
        "Rank": tucker_rank,
        "R2X": tucker_R2X,
        "Method": ["Tucker"] * len(tucker_R2X)
    })


    # visualize core tensor
    core_tensor = pd.DataFrame({
        "Index": product(*[np.arange(1, si + 1) for si in tres[0].shape]),
        "Weight": tres[0].flatten(),
        "Fraction": tres[0].flatten() ** 2,
        "Sign": np.sign(tres[0].flatten()),
    })
    core_tensor = core_tensor.sort_values("Fraction", ascending=False)
    core_tensor["Index"] = core_tensor["Index"].astype(str)
    core_tensor["Fraction"] = core_tensor["Fraction"] / np.sum(core_tensor["Fraction"])

    sns.barplot(core_tensor.iloc[:12, :], ax=ax[0], x="Index", y="Fraction", color="Sign",
                palette=["#2369BD", "#A9393C"])
    ax[0].tick_params(axis='x', rotation=90)

    # visualize factors
    factors = [pd.DataFrame(tres[1][rr],
                            columns=[f"Cmp. {i}" for i in np.arange(1, tres[1][rr].shape[1] + 1)],
                            index=CoH_Data.coords[CoH_Data.coords.dims[rr]].to_numpy())
               for rr in range(data.ndim)]

    for rr in range(data.ndim):
        if CoH_Data.coords.dims[rr] == "Patient":
            sns.clustermap(factors[rr], col_cluster=False, method='average',
                           cmap="vlag", vmin=-1.0, vmax=1.0, cbar=True, ax=ax[rr + 1])
        else:
            sns.heatmap(factors[rr], cmap="vlag", center=0,
                        xticklabels=[str(ii + 1) for ii in range(tres[1][rr].shape[1])],
                        yticklabels=factors[rr].index,
                        cbar=True, vmin=-1.0, vmax=1.0, ax=ax[rr + 1])
        ax[rr + 1].set_xlabel("Components")
        ax[rr + 1].set_title(CoH_Data.coords.dims[rr])

    return f
    f.savefig("new_fig8.pdf")
