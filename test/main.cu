#include <iostream>
#include <algorithm>

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

void applyFilter(const std::string& path, const std::string& output, const cnv::Kernel& kernel) {

    int width, height, bpp;
    unsigned char* buff = stbi_load(path.c_str(), &width, &height, &bpp, STBI_rgb_alpha);

    cnv::Matrix<int> R(width, height);
    cnv::Matrix<int> G(width, height);
    cnv::Matrix<int> B(width, height);

    for(int i = 0; i < width; i ++) {
        for(int j = 0; j < height; j ++) {

            unsigned char r = buff[4 * (i + j * width)];
            R.set(static_cast<int>(r), i, j);

            unsigned char g = buff[4 * (i + j * width) + 1];
            G.set(static_cast<int>(g), i, j);

            unsigned char b = buff[4 * (i + j * width) + 2];
            B.set(static_cast<int>(b), i, j);
        }
    }

    stbi_image_free(buff);

    cnv::Matrix<int> outputR(width, height);
    cnv::Matrix<int> outputG(width, height);
    cnv::Matrix<int> outputB(width, height);

    conv2D(R, outputR, kernel);
    conv2D(G, outputG, kernel);
    conv2D(B, outputB, kernel);

    unsigned char* outputBuff = new unsigned char[width * height * 3];

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {

            int r = outputR.get(i, j);
            int g = outputG.get(i, j);
            int b = outputB.get(i, j);

            outputBuff[3 * (i + j * width)] = static_cast<unsigned char>(std::clamp(r, 0, 255));
            outputBuff[3 * (i + j * width) + 1] = static_cast<unsigned char>(std::clamp(g, 0, 255));
            outputBuff[3 * (i + j * width) + 2] = static_cast<unsigned char>(std::clamp(b, 0, 255));
        }
    }

    stbi_write_png(output.c_str(), width, height, STBI_rgb, outputBuff, width * STBI_rgb);
    delete[] outputBuff;
}

int main() {

    testConvolution();

    cnv::Kernel filterBoxBlur = {
        { 1.0/9, 1.0/9, 1.0/9 },
        { 1.0/9, 1.0/9, 1.0/9 },
        { 1.0/9, 1.0/9, 1.0/9 }
    };

    applyFilter("res/orloj.png", "outputBoxBlur.png", filterBoxBlur);

    cnv::Kernel filterLaplacian = {
        {  0, -1,  0 },
        { -1,  4,  0 },
        {  0, -1,  0 }
    };

    applyFilter("res/orloj.png", "outputLaplacian.png", filterLaplacian);

    cnv::Kernel filterSharpen = {
        {  0, -1,  0 },
        { -1,  5,  0 },
        {  0, -1,  0 }
    };

    applyFilter("res/orloj.png", "outputSharpen.png", filterSharpen);

    cnv::Kernel filterGaussian = {
        { 1.f / 16, 2.f / 16, 1.f / 16 },
        { 2.f / 16, 4.f / 16, 2.f / 16 },
        { 1.f / 16, 2.f / 16, 1.f / 16 }
    };

    applyFilter("res/orloj.png", "outputGaussianBlur.png", filterGaussian);

    cnv::Kernel filterSobelH = {
        { -1, 0, 1 },
        { -2, 0, 2 },
        { -1, 0, 1 }
    };

    applyFilter("res/orloj.png", "outputSobelH.png", filterSobelH);

    cnv::Kernel filterSobelV = {
        { 1 , 2 , 1 },
        { 0 , 0 , 0 },
        {-1, -2, -1 }
    };

    applyFilter("res/orloj.png", "outputSobelV.png", filterSobelV);

    return 0;
}