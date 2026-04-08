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
