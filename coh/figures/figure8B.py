from .figure8A import explore_tucker_rank, visualize_tucker

import numpy as np
from tensorly.decomposition import tucker

from ..flow_rec import make_CoH_Tensor_rec
from .common import getSetup, subplotLabel

def makeFigure():
    """ Tucker decomposition figure """
    ax, f = getSetup((8, 8), (2, 2))
    subplotLabel(ax)

    CoH_Data_R = make_CoH_Tensor_rec()
    data = CoH_Data_R.to_numpy()
    mask = np.isfinite(data)

    tres = tucker(np.nan_to_num(data), rank=(8, 8, 9), mask=mask, svd='randomized_svd')
    R2X = 1 - np.nansum((data - tres.to_tensor()) ** 2) / np.nansum(data ** 2)
    print(f"Tucker R2X is {R2X}")
    # CP 12 comp = 0.8839742868088709

    # visualize core tensor
    visualize_tucker(ax, tres, CoH_Data_R)

    return f
    f.savefig("new_fig8b.pdf")
