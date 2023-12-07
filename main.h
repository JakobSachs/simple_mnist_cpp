#ifndef MAIN_H
#define MAIN_H

#include "Eigen/Dense"
#include <vector>

typedef Eigen::VectorXd vec;
typedef Eigen::MatrixXd mat;

/**
 * Applies the ReLU activation function to a vector.
 *
 * @param x The input vector.
 * @return A vector where the ReLU function has been applied to each element.
 */
vec activation_relu(vec &x);

/**
 * Computes the derivative of the ReLU function for backpropagation.
 *
 * @param x The input vector.
 * @return A vector of derivatives.
 */
vec derive_activation_relu(vec &x);

/**
 * Applies the softmax activation function to a vector.
 *
 * This function is typically used in the output layer of a network to
 * convert the output into a probability distribution.
 *
 * @param x The input vector.
 * @return A vector where the softmax function has been applied to each element.
 */
vec activation_softmax(vec &x);

/**
 * Computes the derivative of the softmax function for backpropagation.
 *
 * @param x The input vector.
 * @return A vector of derivatives.
 */
vec derive_activation_softmax(vec &x);

/**
 * Performs the feed-forward operation for a single layer of the neural network.
 *
 * @param x The input vector.
 * @param w The weight matrix.
 * @param b The bias vector.
 * @return The output vector after applying the weight matrix and bias.
 */
vec feed_forward(vec &x, mat &w, vec &b);

/**
 * Performs the back-propagation step in training the neural network.
 *
 * @param x The input vector.
 * @param w The weight matrix.
 * @param b The bias vector.
 * @param t The target output vector.
 * @return The delta vector used for adjusting the weights and biases.
 */
vec back_propagation(vec &x, mat &w, vec &b, vec &t);

/**
 * Initializes weights and biases for the neural network layers.
 *
 * Weights are initialized with small random values, and biases are initialized
 * to zero.
 *
 * @param w1 The weight matrix for the first layer.
 * @param b1 The bias vector for the first layer.
 * @param w2 The weight matrix for the second layer.
 * @param b2 The bias vector for the second layer.
 * @param input_size The size of the input layer.
 * @param hidden_size The size of the hidden layer.
 * @param output_size The size of the output layer.
 */
void initialize_weights_and_biases(mat &w1, vec &b1, mat &w2, vec &b2,
                                   int input_size, int hidden_size,
                                   int output_size);

/**
 * Computes the cross-entropy loss for a prediction.
 *
 * @param t The target output vector.
 * @param y The predicted output vector.
 * @return The computed loss value.
 */
double compute_loss(const vec &t, const vec &y);

/**
 * Performs a single training step on a given image and label.
 *
 * @param image The input image as a matrix.
 * @param label The target label.
 * @param w1 The weight matrix for the first layer.
 * @param b1 The bias vector for the first layer.
 * @param w2 The weight matrix for the second layer.
 * @param b2 The bias vector for the second layer.
 * @param learning_rate The learning rate for weight and bias updates.
 * @return The loss after the training step.
 */
double train_step(const mat &image, const vec &label, mat &w1, vec &b1, mat &w2,
                  vec &b2, double learning_rate);

/**
 * Performs a training step for a batch of images and labels.
 *
 * This function optimizes the neural network parameters using a batch of data
 * instead of a single data point, which often results in more stable and faster
 * training.
 *
 * @param batch_images A vector of image matrices.
 * @param batch_labels A vector of corresponding labels.
 * @param w1 The weight matrix for the first layer.
 * @param b1 The bias vector for the first layer.
 * @param w2 The weight matrix for the second layer.
 * @param b2 The bias vector for the second layer.
 * @param learning_rate The learning rate for weight and bias updates.
 * @return The average loss over the batch.
 */
double train_batch_step(const std::vector<mat> &batch_images,
                        const std::vector<int> &batch_labels, mat &w1, vec &b1,
                        mat &w2, vec &b2, double learning_rate);

#endif // MAIN_H
