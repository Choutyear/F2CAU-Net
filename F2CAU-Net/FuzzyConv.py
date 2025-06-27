import torch
import torch.nn.functional as F


def fuzzy_convolution(input, fuzzy_filter):
    #
    expanded_input = input.unsqueeze(0)
    expanded_filter = fuzzy_filter.unsqueeze(0).unsqueeze(0)


    output = F.conv2d(expanded_input, expanded_filter, stride=1, padding=0)

    return output


input = torch.randn(1, 1, 5, 5)  # [batch_size, channels, height, width]
fuzzy_filter = torch.tensor([[0.2, 0.4, 0.2],
                             [0.4, 0.6, 0.4],
                             [0.2, 0.4, 0.2]])


output = fuzzy_convolution(input, fuzzy_filter)


print(output)
