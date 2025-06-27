# F2CAU-Net
F2CAU-Net: A Dual Fuzzy Medical Image Segmentation Cascade Method Based on Fuzzy Feature Learning

We are currently updating the codebase, including revising comments and reorganizing the folder structure. A reference document for running the code is also in progress. In the meantime, feel free to explore the code directly in the repository. If you have any questions, please contact us at [choutyear@outlook.com](mailto:choutyear@outlook.com)
.

<br>

![image](https://github.com/Choutyear/F2CAU-Net/blob/main/Figs/F1.jpg)

F2CAU-Net model architecture. We introduce fuzzy convolution modules (blue rectangles) to enhance the convolutional neural network’s ability to model fuzzy boundaries and regions. In addition, a fuzzy attention mechanism (purple rectangle) is incorporated into the skip connection to optimize the segmentation results.

<br>

![image](https://github.com/Choutyear/F2CAU-Net/blob/main/Figs/F2.jpg)

Structure of the fuzzy convolution layer. The module consists of two parallel branches: (1) a standard convolutional path that extracts basic feature maps through two 3×3 convolutions; (2) a fuzzy logic path where each pixel is processed by Gaussian membership functions to obtain fuzzy activations.

<br>

![image](https://github.com/Choutyear/F2CAU-Net/blob/main/Figs/F3.jpg)

Fuzzy attention mechanism structure. We combine the fuzzy logic method with the attention mechanism, using Gaussian membership function and adding voxel by voxel to fuse information, and then input the obtained feature representation into fuzzy attention to generate an attention feature map in the voxel direction.

