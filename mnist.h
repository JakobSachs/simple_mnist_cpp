#ifndef MNIST_H
#define MNIST_H
#include <bit>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "main.h"

/**
 * Reads and loads the MNIST dataset from binary files.
 *
 * This function handles both the training and testing sets of the MNIST
 * database, which consists of handwritten digit images and their corresponding
 * labels. It reads the dataset's binary files, extracts the image and label
 * data, and stores them in the provided vectors.
 *
 * @param trainset Selects the training set (true) or the test set (false).
 * @param images Vector to store the images as matrices.
 * @param labels Vector to store the corresponding labels.
 * @return True if successful, False if file reading fails.
 */
bool read_mnist_data(bool trainset, std::vector<mat> &images,
                     std::vector<int> &labels) {
  // Construct filename
  std::string filename = trainset ? "mnist/train-images-idx3-ubyte"
                                  : "mnist/t10k-images-idx3-ubyte";

  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return false;
  }

  // Read header information as done in the original code...
  int magic_number = 0;
  file.read((char *)&magic_number, sizeof(magic_number));
  magic_number = std::byteswap(magic_number);

  if (magic_number != 2051) {
    std::cerr << "Invalid MNIST image file!" << std::endl;
    return false;
  }

  int number_of_images = 0;
  file.read((char *)&number_of_images, sizeof(number_of_images));
  number_of_images = std::byteswap(number_of_images);

  int number_of_rows = 0;
  file.read((char *)&number_of_rows, sizeof(number_of_rows));
  number_of_rows = std::byteswap(number_of_rows);

  int number_of_columns = 0;
  file.read((char *)&number_of_columns, sizeof(number_of_columns));
  number_of_columns = std::byteswap(number_of_columns);

  assert(number_of_images > 0);
  assert(number_of_rows > 0);
  assert(number_of_columns > 0);

  // Read image data into `images` vector
  for (int i = 0; i < number_of_images; ++i) {
    mat image(number_of_rows, number_of_columns);
    unsigned char pixel[1];
    for (int j = 0; j < number_of_rows * number_of_columns; ++j) {
      file.read((char *)&pixel, 1);
      image(j / number_of_columns, j % number_of_columns) = pixel[0] / 255.0;
    }
    images.push_back(image);
  }

  // Close file
  file.close();

  // Construct filename for label data
  filename = trainset ? "mnist/train-labels-idx1-ubyte"
                      : "mnist/t10k-labels-idx1-ubyte";

  // Open file
  file.open(filename, std::ios::binary);

  // Read header information as done in the original code...
  file.read((char *)&magic_number, sizeof(magic_number));
  magic_number = std::byteswap(magic_number);

  if (magic_number != 2049) {
    std::cerr << "Invalid MNIST label file!" << std::endl;
    return false;
  }

  int number_of_labels = 0;
  file.read((char *)&number_of_labels, sizeof(number_of_labels));
  number_of_labels = std::byteswap(number_of_labels);

  assert(number_of_labels == number_of_images);

  // Read label data into `labels` vector
  for (int i = 0; i < number_of_labels; ++i) {
    unsigned char label[1];
    file.read((char *)&label, 1);
    labels.push_back(label[0]);
  }

  return true; // Return true if successful
}

#endif // MNIST_H
