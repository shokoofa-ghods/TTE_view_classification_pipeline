"""
Data visualization module
"""

import matplotlib.pyplot as plt
import torch
import seaborn as sns
import os

def visualization(image:torch.Tensor, label:torch.Tensor, dir:str):
    """
    Plot an image with label
    
    Args:
        image (Image): The image to display
        label (str): The title for the image
    """
    plt.figure(figsize= (8,8))
    # im = Image.fromarray(image.permute(1,2,0))
    # plt.imshow(image.permute(1,2,0))
    plt.imshow(image, cmap= 'gray')
    plt.title(str(label.item()))
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(dir,'image_sample.png'))

def show_confusion_matrix(cm,
                          labels_name,
                          dir: str,
                          title:str='Confusion Matrix',
                          x_label:str='Predicted Labels',
                          y_label:str='True Labels',
                          fmt:str='.2g',
                          cmap:str='Blues',
                          figsize:tuple[int, int]=(10, 7)) -> None:
    """
    Plot and show a confusion matrix along with its labels

    Args:
        cm (_ArrayLike[Incomplete] | DataFrame): The confusion matrix
        labels_name (Sequence[str]): The sequence of string labels 
    """
    plt.figure(figsize=figsize)
    sns.heatmap(cm,
                annot=True,
                fmt=fmt,
                cmap=cmap,
                xticklabels=labels_name,
                yticklabels=labels_name )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(os.path.join(dir, 'confusion_matrix.png'))
