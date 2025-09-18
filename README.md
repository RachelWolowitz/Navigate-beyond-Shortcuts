---
license: mit
---
# Navigate Beyond Shortcuts: Debiased Learning through the Lens of Neural Collapse
[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of ["Navigate beyond shortcuts: Debiased learning through the lens of neural collapse"](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Navigate_Beyond_Shortcuts_Debiased_Learning_Through_the_Lens_of_Neural_CVPR_2024_paper.html).

## Overview

Recent studies have noted an intriguing phenomenon termed Neural Collapse, that is, when the neural networks establish the right correlation between feature spaces and the training targets, their last-layer features, together with the classifier weights, will collapse into a stable and symmetric structure. In this paper, we extend the investigation of Neural Collapse to the biased datasets with imbalanced attributes. We observe that models will easily fall into the pitfall of shortcut learning and form a biased, non-collapsed feature space at the early period of training, which is hard to reverse and limits the generalization capability. To tackle the root cause of biased classification, we follow the recent inspiration of prime training, and propose an avoid-shortcut learning framework without additional training complexity. With well-designed shortcut primes based on Neural Collapse structure, the models are encouraged to skip the pursuit of simple shortcuts and naturally capture the intrinsic correlations. Experimental results demonstrate that our method induces better convergence properties during training, and achieves state-of-the-art generalization performance on both synthetic and real-world biased datasets.

## Usage
We provide the code files for ETF-Debias training. The main implementation is in `train_ETF.py`.

The **recommended usage** is as follows:

1. Prepare the environment.
```
cd Navigate-beyond-Shortcuts
pip install -r requirements.txt
```
2. We provide our datasets for evaluation in ``./data``, including 2 synthetic datasets (i.e., ColoredMNIST, CorruptedCIFAR10), and 3 real-world biased datasets (i.e., Biased FFHQ (BFFHQ), BAR, Dogs & Cats). 

3. Run ETF-Debias training:
```
python train_ETF.py
```

## Citation
```
@inproceedings{wang2024navigate,
  title={Navigate beyond shortcuts: Debiased learning through the lens of neural collapse},
  author={Wang, Yining and Sun, Junjie and Wang, Chenyue and Zhang, Mi and Yang, Min},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12322--12331},
  year={2024}
}
```
## Acknowledgement
This repo is based on the codebase of [LfF](https://github.com/alinlab/LfF/tree/master). We sincerely thank the contributors of their valuable work.