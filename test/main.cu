#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "vendor/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vendor/stb_image_write.h"

#include "../src/conv.cuh"

void test() {

    cnv::Matrix<int> input = {
        { 1, 1, 1, 1, 1 },
        { 1, 4, 5, 4, 1 },
        { 1, 5, 6, 5, 1 },
        { 1, 4, 5, 4, 1 },
        { 1, 1, 1, 1, 1 }
    };

    std::cout << "Input matrix\n" << input << std::endl;

    cnv::Kernel laplacian = {
        { 0,  1, 0 },
        { 1, -4, 1 },
        { 0,  1, 0 }
    };

    std::cout << "Kernel\n" << laplacian << std::endl;

    cnv::Matrix<int> output(input.rows, input.cols);
    conv2D(input, output, laplacian);

    std::cout << "Output matrix\n" << output << std::endl;
}

int main() {

    test();

    return 0;
}