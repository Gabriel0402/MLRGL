# Tensorial multiview low-rank high-order graph learning for context-enhanced domain adaptation

## Abstract

Unsupervised Domain Adaptation (UDA) is a machine learning technique that facilitates knowledge transfer from a labeled source domain to an unlabeled target domain, addressing distributional discrepancies between these domains. Existing UDA methods often fail to effectively capture and utilize contextual relationships within the target domain. This research introduces a novel framework called Tensorial Multiview Low-Rank High-Order Graph Learning (MLRGL), which addresses these challenges by learning high-order graphs constrained by low-rank tensors to uncover contextual relations. The proposed framework ensures prediction consistency between randomly masked target images and their pseudo-labels by leveraging spatial context to generate multiview domain-invariant features through various augmented masking techniques. A high-order graph is constructed by combining Laplacian graphs to propagate these multiview features. Low-rank constraints are applied along both horizontal and vertical dimensions to better uncover inter-view and inter-class correlations among multiview features. This high-order graph is used to create an affinity matrix, mapping multiview features into a unified subspace. Prototype vectors and unsupervised clustering are then employed to calculate conditional probabilities for UDA tasks. We evaluated our approach using three different backbones across three benchmark datasets. The results demonstrate that the MLRGL framework outperforms current state-of-the-art methods in various UDA tasks. Additionally, our framework exhibits robustness to hyperparameter variations and demonstrates that multiview approaches outperform single-view solutions.


# How to use the code
You need to change the directory to your data.
You can download extracted features used in our experiments from [BaiduPan](https://pan.baidu.com/s/1g3DmZcFMAKzHS9ihAZpysQ) code: r7ht\
zcy@cczu.edu.cn
# Reference

Zhu, C., Zhang, L., Luo, W., Jiang, G., & Wang, Q. (2024). Tensorial multiview low-rank high-order graph learning for context-enhanced domain adaptation. Neural Networks, 106859.