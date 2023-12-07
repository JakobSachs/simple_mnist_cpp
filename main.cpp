/**
 * * This is a simple implementation of a neural network to learn MNIST.
 * Author: Jakob Sachs
 * **/

#include "main.h"
#include "mnist.h"
#include <cassert>
#include <iostream>
#include <random>

vec activation_relu(vec &x) {
  vec y = x;
  for (int i = 0; i < x.size(); i++) {
    y(i) = std::max(0.0, x(i));
  }

  return y;
}

// derive of activation function relu
vec derive_activation_relu(vec &x) {
  vec y = x;
  for (int i = 0; i < x.size(); i++) {
    if (x(i) > 0) {
      y(i) = 1;
    } else {
      y(i) = 0;
    }
  }

  return y;
}

// acitvation function softmax
vec activation_softmax(vec &x) {
  double maxVal = x.maxCoeff();
  vec exp_x = (x.array() - maxVal).exp();
  return exp_x / exp_x.sum();
}

// derive of activation function softmax
vec derive_activation_softmax(vec &x) {
  vec y = x;
  for (int i = 0; i < x.size(); i++) {
    y(i) = x(i) * (1 - x(i));
  }

  return y;
}

// feed forward
vec feed_forward(vec &x, mat &w, vec &b) {
  assert(x.size() == w.cols());
  vec y = w * x + b;
  return y;
}

// back propagation
vec back_propagation(vec &x, mat &w, vec &b, vec &t) {
  vec y = feed_forward(x, w, b);
  vec delta = derive_activation_relu(y).cwiseProduct(t - y);
  return delta;
}

// Utility function to initialize weights and biases

void initialize_weights_and_biases(mat &w1, vec &b1, mat &w2, vec &b2,
                                   int input_size, int hidden_size,
                                   int output_size) {
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0, 0.01);

  w1 = mat::NullaryExpr(hidden_size, input_size,
                        [&]() { return distribution(generator); });
  b1 = vec::Zero(hidden_size);
  w2 = mat::NullaryExpr(output_size, hidden_size,
                        [&]() { return distribution(generator); });
  b2 = vec::Zero(output_size);
}

// Utility function to compute cross entropy loss
double compute_loss(const vec &t, const vec &y) {
  constexpr double epsilon = 1e-7;
  double loss = 0.0;
  for (int i = 0; i < t.size(); ++i) {
    loss -= t(i) * std::log(y(i) + epsilon);
  }
  return loss;
}

// Modularized training step
double train_step(const mat &image, const vec &label, mat &w1, vec &b1, mat &w2,
                  vec &b2, double learning_rate) {
  // Convert image to a vector
  vec x = Eigen::Map<const vec>(image.data(), image.size());

  // Forward pass

  vec y1 = feed_forward(x, w1, b1);
  y1 = activation_relu(y1);
  vec y2 = feed_forward(y1, w2, b2);
  y2 = activation_softmax(y2);

  // Backward pass (compute gradients)
  vec delta2 = derive_activation_softmax(y2).cwiseProduct(label - y2);
  vec delta1 = derive_activation_relu(y1).cwiseProduct(w2.transpose() * delta2);

  // Update weights and biases
  w2 += learning_rate * delta2 * y1.transpose();
  b2 += learning_rate * delta2;
  w1 += learning_rate * delta1 * x.transpose();
  b1 += learning_rate * delta1;

  // calc loss
  return compute_loss(label, y2);
}

// Modularized training step for a mini-batch
double train_batch_step(const std::vector<mat> &batch_images,
                        const std::vector<int> &batch_labels, mat &w1, vec &b1,
                        mat &w2, vec &b2, double learning_rate) {
  // Initialize accumulators for gradients
  mat grad_w1 = mat::Zero(w1.rows(), w1.cols());
  vec grad_b1 = vec::Zero(b1.size());
  mat grad_w2 = mat::Zero(w2.rows(), w2.cols());
  vec grad_b2 = vec::Zero(b2.size());

  double batch_loss = 0.0;

#pragma omp parallel for shared(batch_loss, grad_w1, grad_b1, grad_w2, grad_b2)
  for (size_t i = 0; i < batch_images.size(); ++i) {
    // Convert image to a vector
    vec x =
        Eigen::Map<const vec>(batch_images[i].data(), batch_images[i].size());

    // Create one-hot vector for label
    vec t = vec::Zero(10);
    t(batch_labels[i]) = 1;

    // Forward pass
    vec y1 = feed_forward(x, w1, b1);
    y1 = activation_relu(y1);
    vec y2 = feed_forward(y1, w2, b2);
    y2 = activation_softmax(y2);

    // Backward pass (compute gradients)
    vec delta2 = derive_activation_softmax(y2).cwiseProduct(t - y2);
    vec delta1 =
        derive_activation_relu(y1).cwiseProduct(w2.transpose() * delta2);

    // Accumulate gradients
    grad_w2 += delta2 * y1.transpose();
    grad_b2 += delta2;
    grad_w1 += delta1 * x.transpose();
    grad_b1 += delta1;

    // Accumulate loss
    batch_loss += compute_loss(t, y2);
  }

  // Average gradients and loss
  grad_w1 /= batch_images.size();
  grad_b1 /= batch_images.size();
  grad_w2 /= batch_images.size();
  grad_b2 /= batch_images.size();
  batch_loss /= batch_images.size();

  // Update weights and biases with momentum
  w1 += learning_rate * grad_w1;
  b1 += learning_rate * grad_b1;
  w2 += learning_rate * grad_w2;
  b2 += learning_rate * grad_b2;

  // Optionally, you can return the average loss of the batch
  return batch_loss;
}

// run a simple neural network to learn MNIST
int main() {
  // READ TRAINING DATA
  std::vector<mat> images;
  std::vector<int> labels;

  // read image data
  bool success = read_mnist_data(true, images, labels);
  if (!success) {
    std::cout << "Failed to read training data!" << std::endl;
    return 0;
  }

  // print first image
  std::cout << "first image:" << std::endl;
  mat image = images[0];
  for (int i = 0; i < image.rows(); i++) {
    for (int j = 0; j < image.cols(); j++) {
      if (image(i, j) > 0.) {
        std::cout << "# ";
      } else {
        std::cout << "_ ";
      }
    }
    std::cout << std::endl;
  }

  // setup neural network
  int input_size = 28 * 28;
  int hidden_size = 100;
  int output_size = 10;

  // initialize weight and bias
  mat w1, w2;
  vec b1, b2;
  initialize_weights_and_biases(w1, b1, w2, b2, input_size, hidden_size,
                                output_size);

  // learning rate and batch size
  double learning_rate = 0.025;
  size_t batch_size = 64;
  size_t epochs = 15;

  // train neural network on image
  for (size_t epoch = 0; epoch < epochs; epoch++) {
    int idx = 0;
    while (idx * batch_size < images.size()) {
      vec t = vec::Zero(output_size);
      t(labels[idx]) = 1;

      // double loss = train_step(image, t, w1, b1, w2, b2, learning_rate);
      size_t start = idx * batch_size;
      size_t end = std::min(start + batch_size, images.size());

      std::vector<mat> batch_images(images.begin() + start,
                                    images.begin() + end);
      std::vector<int> batch_labels(labels.begin() + start,
                                    labels.begin() + end);

      double loss = train_batch_step(batch_images, batch_labels, w1, b1, w2, b2,
                                     learning_rate);

      // print progress
      if (idx % 200 == 0) {
        std::cout << "epoch: " << epoch << ", batch: " << idx << "/"
                  << images.size() / batch_size << ", loss: " << loss
                  << std::endl;
      }
      idx++;
    }
  }

  // READ VALIDATION DATA
  std::vector<mat> images_validation;
  std::vector<int> labels_validation;

  // read image data
  success = read_mnist_data(false, images_validation, labels_validation);
  if (!success) {
    std::cout << "Failed to read validation data!" << std::endl;
    return 0;
  }

  // validate neural network on image
  int correct = 0;
  for (size_t idx = 0; idx < images_validation.size(); idx++) {
    vec x = Eigen::Map<const vec>(images_validation[idx].data(),
                                  images_validation[idx].size());

    vec y1 = feed_forward(x, w1, b1);
    y1 = activation_relu(y1);
    vec y2 = feed_forward(y1, w2, b2);
    y2 = activation_softmax(y2);

    int label = std::distance(
        y2.data(), std::max_element(y2.data(), y2.data() + y2.size()));
    if (label == labels_validation[idx]) {
      correct++;
    }
  }

  printf("accuracy: %f\n", (double)correct / images_validation.size());
  return 0;
}
