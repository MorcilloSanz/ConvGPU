#include <iostream>
#include <cstdlib>

#define STB_IMAGE_IMPLEMENTATION
#include "vendor/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vendor/stb_image_write.h"

#include "../src/conv.cuh"

void testConvolution() {

    cnv::Matrix<int> input = {
        { 1, 1, 1, 1, 1 },
        { 1, 4, 5, 4, 1 },
        { 1, 5, 6, 5, 1 },
        { 1, 3, 5, 4, 1 },
        { 1, 1, 1, 1, 1 }
    };

    std::cout << "Input matrix\n" << input << std::endl;

    cnv::Kernel laplacian = {
        { 0,  1, 0 },
        { 1, -4, 1 },
        { 0,  1, 0 }
    };

    std::cout << "Kernel\n" << laplacian << std::endl;

    cnv::Matrix<int> output(input.cols, input.rows);
    conv2D(input, output, laplacian);

    std::cout << "Output matrix\n" << output << std::endl;
}

void applyFilter(const std::string& path, const cnv::Kernel& kernel) {

    int width, height, bpp;
    unsigned char* buff = stbi_load(path.c_str(), &width, &height, &bpp, STBI_rgb_alpha);

    cnv::Matrix<unsigned char> R(width, height);
    cnv::Matrix<unsigned char> G(width, height);
    cnv::Matrix<unsigned char> B(width, height);

    for(int i = 0; i < width; i ++) {
        for(int j = 0; j < height; j ++) {

            unsigned char r = buff[4 * (i + j * width)];
            R.set(r, i, j);

            unsigned char g = buff[4 * (i + j * width) + 1];
            G.set(g, i, j);

            unsigned char b = buff[4 * (i + j * width) + 2];
            B.set(b, i, j);
        }
    }

    stbi_image_free(buff);

    cnv::Matrix<unsigned char> outputR(width, height);
    cnv::Matrix<unsigned char> outputG(width, height);
    cnv::Matrix<unsigned char> outputB(width, height);

    conv2D(R, outputR, kernel);
    conv2D(G, outputG, kernel);
    conv2D(B, outputB, kernel);

    unsigned char* outputBuff = (unsigned char*)malloc(sizeof(unsigned char) * width * height * 3);

    for(int i = 0; i < width; i ++) {
        for(int j = 0; j < height; j ++) {
            outputBuff[3 * (i + j * width)] = outputR.get(i, j);
            outputBuff[3 * (i + j * width) + 1] = outputG.get(i, j);
            outputBuff[3 * (i + j * width) + 2] = outputB.get(i, j);
        }
    }

    stbi_write_png("output.png", width, height, STBI_rgb, outputBuff, width * STBI_rgb);
    free(outputBuff);
}

int main() {

    testConvolution();

    cnv::Kernel filter = {
        { 0.f,  -1.f,  0.f },
        { -1.f,  5.f, -1.f },
        { 0.f,  -1.f,  0.f }
    };

    applyFilter("res/orloj.png", filter);

    return 0;
}