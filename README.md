# ConvolutionGPU
This header file defines the necessary structures and functions to perform 2D discrete convolution on matrices utilizing CUDA for parallel processing. It includes a matrix class template for managing 2D data and functions for performing convolution and other related operations.

The discrete 2D convolution is defined as:

$$
g(x, y) = \omega \ast f(x, y) = \sum_{i=-a}^{a} \sum_{j=-b}^{b} \omega(i, j) f(x - i, y - j),
$$

where $g(x, y)$ is the filtered image, $f(x, y)$ is the original image, and $\omega$ is the filter kernel. Every element of the filter kernel is considered within the range $-a \leq i \leq a$ and $-b \leq j \leq b$.
