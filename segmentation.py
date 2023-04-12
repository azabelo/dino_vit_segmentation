import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from dino.cosegmentation import find_cosegmentation, draw_cosegmentation, draw_cosegmentation_binary_masks
import sys


#@title Configuration:
#@markdown Choose image paths:
images_paths = ['dino/images/cat.jpg', 'dino/images/ibex.jpg'] #@param
#@markdown Choose loading size:
load_size = 360 #@param
#@markdown Choose layer of descriptor:
layer = 11 #@param
#@markdown Choose facet of descriptor:
facet = 'key' #@param
#@markdown Choose if to use a binned descriptor:
bin=False #@param
#@markdown Choose fg / bg threshold:
thresh=0.065 #@param
#@markdown Choose model type:
model_type='dino_vits8' #@param
#@markdown Choose stride:
stride=4 #@param
#@markdown Choose elbow coefficient for setting number of clusters
elbow=0.975 #@param
#@markdown Choose percentage of votes to make a cluster salient.
votes_percentage=75 #@param
#@markdown Choose whether to remove outlier images
remove_outliers=False #@param
#@markdown Choose threshold to distinguish inliers from outliers
outliers_thresh=0.7 #@param
#@markdown Choose interval for sampling descriptors for training
sample_interval=100 #@param
#@markdown Use low resolution saliency maps -- reduces RAM usage.
low_res_saliency_maps=True #@param


with torch.no_grad():
    # computing cosegmentation
    seg_masks, pil_images = find_cosegmentation(images_paths, elbow, load_size, layer, facet, bin, thresh, model_type,
                                                stride, votes_percentage, sample_interval, remove_outliers,
                                                outliers_thresh, low_res_saliency_maps)

    figs, axes = [], []
    for pil_image in pil_images:
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(pil_image)
        figs.append(fig)
        axes.append(ax)

    # saving cosegmentations
    binary_mask_figs = draw_cosegmentation_binary_masks(seg_masks)
    chessboard_bg_figs = draw_cosegmentation(seg_masks, pil_images)

    plt.show()









import CUB_200_2011.CUB_200_2011.images as images

print(sys.version)

images_path = "CUB_200_2011/CUB_200_2011/images"
for category_name in os.listdir(images_path):
    print(category_name)
    category_path = os.path.join(images_path,category_name)
    for image_name in os.listdir(category_path):
        print(image_name)
        image_path = os.path.join(category_path,image_name)
        img = mpimg.imread(image_path)
        plt.imshow(img)
        plt.show()