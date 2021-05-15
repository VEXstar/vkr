#from vkr.loader.ct_loader import load_data_as_data_loader
import vkr.recognizers.predictor as pr
import os
import matplotlib.pyplot as plt
from io import StringIO
import numpy as np


def get_img(arr):
    fig = plt.figure()
    plt.imshow(arr)

    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)

    return imgdata.getvalue()


def get_masks(data_loader):
    masks = pr.predict(data_loader)

    return masks


# def do_analyzing(f_path):
#     ct_dl = load_data_as_data_loader(f_path, 160, 1)
#     os.remove(f_path)
#     masks = get_masks(ct_dl)
#     #TODO work here
#     return masks



# def deco_do_analyzing(f_path):
#     ct_dl = load_data_as_data_loader(f_path, 160, 1)
#     os.remove(f_path)
#     scan = []
#     mask = []
#     masks = get_masks(ct_dl)
#     i = 0
#     for x in ct_dl:
#         if len(np.unique(masks[i][0][0])) < 2:
#             i = i + 1
#             continue
#         scan.append(get_img(x[0][0][0]))
#         mask.append(get_img(masks[i][0][0]))
#         i = i + 1
#         if len(masks) <= i:
#             break
#     return {"mask": mask, "scan": scan}
