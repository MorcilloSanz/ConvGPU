#include <iostream>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "vendor/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vendor/stb_image_write.h"

#include <conv.cuh>

/**
 * @brief Tests the 2D convolution function with a sample input matrix and kernel.
 *
 * This function creates a sample 5x5 input matrix and a Laplacian kernel,
 * then performs a 2D convolution on the input matrix with the given kernel.
 * The input matrix, kernel, and output matrix are printed to the console.
 */
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

/**
 * @brief Applies a 2D convolution filter to an image and saves the result.
 *
 * This function loads an image from the specified path, separates the RGB channels,
 * applies the given convolution kernel to each channel, and then writes the filtered
 * image to the output path.
 *
 * @param path The file path to the input image.
 * @param output The file path where the filtered image will be saved.
 * @param kernel The convolution kernel to be applied to the image.
 *
 * The function performs the following steps:
 * - Loads the input image and converts it to RGB format.
 * - Separates the RGB channels into individual matrices.
 * - Applies the given convolution kernel to each RGB channel separately.
 * - Combines the filtered RGB channels and writes the output to a file.
 */
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

    std::cout << "Appying box blur filter" << std::endl;
    applyFilter("res/orloj.png", "boxblur.png", filterBoxBlur);

    cnv::Kernel filterLaplacian = {
        {  0, -1,  0 },
        { -1,  4,  0 },
        {  0, -1,  0 }
    };

    std::cout << "Appying laplacian filter" << std::endl;
    applyFilter("res/orloj.png", "laplacian.png", filterLaplacian);

    cnv::Kernel filterSharpen = {
        {  0, -1,  0 },
        { -1,  5,  0 },
        {  0, -1,  0 }
    };

    std::cout << "Appying sharpen filter" << std::endl;
    applyFilter("res/orloj.png", "sharpen.png", filterSharpen);

    cnv::Kernel filterGaussian = {
        { 1.f / 16, 2.f / 16, 1.f / 16 },
        { 2.f / 16, 4.f / 16, 2.f / 16 },
        { 1.f / 16, 2.f / 16, 1.f / 16 }
    };

    std::cout << "Appying gaussian blur filter" << std::endl;
    applyFilter("res/orloj.png", "gaussianblur.png", filterGaussian);

    cnv::Kernel filterSobelH = {
        { -1, 0, 1 },
        { -2, 0, 2 },
        { -1, 0, 1 }
    };

    std::cout << "Appying horizontal sobel filter" << std::endl;
    applyFilter("res/orloj.png", "sobelh.png", filterSobelH);

    cnv::Kernel filterSobelV = {
        { 1 , 2 , 1 },
        { 0 , 0 , 0 },
        {-1, -2, -1 }
    };

    std::cout << "Appying vertical sobel filter" << std::endl;
    applyFilter("res/orloj.png", "sobelv.png", filterSobelV);

    return 0;
}