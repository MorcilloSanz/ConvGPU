#include <iostream>

#include "../src/conv.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "vendor/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vendor/stb_image_write.h"

void test() {

    cnv::Matrix<int> input = {
        { 1, 1, 1, 1, 1 },
        { 1, 5, 6, 5, 1 },
        { 1, 5, 6, 5, 1 },
        { 1, 1, 1, 1, 1 }
    };

    cnv::Kernel laplacian = {
        { 0,  1, 0 },
        { 1, -4, 1 },
        { 0,  1, 0 }
    };

    cnv::Matrix<int> output(input.rows, input.cols);
    conv2D(input, output, laplacian);

    std::cout << output << std::endl;
}

int main() {

    test();

    return 0;
}