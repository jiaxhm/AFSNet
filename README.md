# Adaptive Fusion of CNN-Transformer for Superpixel Segmentation

Abstract—The convolutional neural network (CNN) and Transformer have powerful feature extraction capabilities. How
ever, CNN employs convolution operations to exploit local features, while Transformer utilizes self-attention mechanisms
to extract global representations. How to effectively combine CNN and Transformer models remains a challenge and is
rarely explored in superpixel segmentation. To address this issue, we propose an Adaptive Fusion of CNN-Transformer
for Superpixel Network (AFSNet). Specifically, AFSNet utilizes an adaptive fusion unit to interactively fuse CNN-derived
local features and Transformer-based global representations. Furthermore, the aggregation decoder integrates adjacent
multi-level features to mine homogeneous attributes. Extensive experiments on diverse sensor datasets demonstrate the
AFSNet significantly outperforms state-of-the-art methods, in terms of boundary adherence, generalization capability, and
visual comparison.

Index Terms—Sensor applications, convolutional neural network (CNN), Transformer, attention, superpixel segmentation.

<img width="2358" height="1445" alt="Fig1" src="https://github.com/user-attachments/assets/baf7d61a-4959-4b05-bb35-2c4a1bd17722" />

To capitalize on the strengths of CNN in capturing local features and Transformer in handling long-range dependencies, we propose AFSNet, as illustrated in Fig. 1. The AFSNet mainly consists of four strategies: (1) Low-level encoder for preserving basic information. (2) High-level encoder for extracting abstract cues. (3) Aggregation decoder for refining multi-scale content. (4) Boundary guided loss for enhancing superpixel regularity.

✅ We have submitted the paper to the IEEE Sensors Letters

✅ We have updated the code

# ✨ Getting Start

# Environment Installation

Reference to SCN (https://github.com/fuy34/superpixel_fcn)and MetaFormer(https://github.com/sail-sg/metaformer).

# Preparing Dataset
1. BSDS500: Following this link: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html
2. NYUDv2: Following this link: http://vcl.ucsd.edu/hed/nyu/
3. KITTI: Following this link: http://www.cvlibs.net/datasets/kitti/

Furthermore, preprocessing of BSDS500 training data follows SCN (https://github.com/fuy34/superpixel_fcn).

# Training
1. The proposed AFSNet is trained on the BSDS500 training set.
2. During training, RGB images and gradient images are used as inputs, and the superpixel loss and edge constraint are jointly optimized.

   Run 'python main.py' to start the program.

   ✨ It is worth mentioning that AFSNet is trained exclusively on the BSDS500 training set and directly generates superpixels for NYUv2 and KITTI without requiring fine-tuning.

# Testing
1. Test BSDS500: Please run `test_bsds.py`.
2. Test NYUDv2: Please run `test_nyu.py`.
3. Test KITTI: Please run `test_kitti.py`.

# Weights
We have placed the pretrained weight model_best.tar in the https://pan.baidu.com/s/19gCcLKVZftkWPwaAm8i1PA password: z4qx

# Acknowledgments

The basic code is partially from the below repos.
1. SCN (https://github.com/fuy34/superpixel_fcn)
2. MetaFormer(https://github.com/sail-sg/metaformer)

# 📚 Cite Us

✨ Please cite us if this work is helpful to you.

```bibtex
@ARTICLE{
  author={Xiaohong Jia, Yonghui Li, Chaoneng Li, and Yao Zhao},
  journal={IEEE Sensors Letters},
  title={Adaptive Fusion of CNN-Transformer for Superpixel Segmentation},
  year={2026},
  volume={},
  pages={},
  keywords={CNN; Transformer; attention; superpixel segmentation},
  doi={}
}
