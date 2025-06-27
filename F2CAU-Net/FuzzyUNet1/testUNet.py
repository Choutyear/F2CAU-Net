import matplotlib.pyplot as plt
import numpy as np

from UNet import *
from FuzzyUNet import *
from FCAUNet import *

# Create an instance of the U-Net model
# model = UNet(num_classes=2)
# model = FuzzyUNet(num_classes=2)
model = FCAUNet(num_classes=2)

# Generate a random image
# image = torch.randn(1, 3, 512, 512)
image = torch.randn(1, 3, 512, 512)

# Test the model on the random image
output = model(image)

# Print the segmentation result
print(output.shape)

# 可视化原始图像和预测的分割结果
# plt.imshow(image[0, 1, :, :])
# plt.show()

# 获取黑白分割结果（选择通道1，即表示黑色的通道）
result = output[0, 1, :, :]

# 将结果转换为灰度图像，确保其形状为 (512, 512)
gray_result = (result * 255).detach().numpy().astype(np.uint8)
plt.imshow(gray_result, cmap='gray')
plt.show()
