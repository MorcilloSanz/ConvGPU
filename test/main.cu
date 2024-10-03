#include <iostream>

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

    int minR = 255, maxR = 0;
    int minG = 255, maxG = 0;
    int minB = 255, maxB = 0;

    for(int i = 0; i < width; i ++) {
        for(int j = 0; j < height; j ++) {

            int r = outputR.get(i, j);
            int g = outputG.get(i, j);
            int b = outputB.get(i, j);

            if(r > maxR) maxR = r;
            else if(r < minR) minR = r;

            if(g > maxG) maxG = g;
            else if(g < minG) minG = g;

            if(b > maxB) maxB = b;
            else if(b < minB) minB = b;
        }
    }

    int incrementR = maxR - minR;
    int incrementG = maxG - minG;
    int incrementB = maxB - minB;

    for(int i = 0; i < width; i ++) {
        for(int j = 0; j < height; j ++) {
            outputBuff[3 * (i + j * width)] =     static_cast<unsigned char>(255 * (outputR.get(i, j) - minR) / incrementR);
            outputBuff[3 * (i + j * width) + 1] = static_cast<unsigned char>(255 * (outputG.get(i, j) - minG) / incrementG);
            outputBuff[3 * (i + j * width) + 2] = static_cast<unsigned char>(255 * (outputB.get(i, j) - minB) / incrementB);
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

    cnv::Kernel filterSharpen = {
        {  0, -1,  0 },
        { -1,  5,  0 },
        {  0, -1,  0 }
    };

    applyFilter("res/orloj.png", "outputSharpen.png", filterSharpen);

    return 0;
}