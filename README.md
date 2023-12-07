# MNIST Neural Network in C++

This project implements a simple neural network in C++ to learn and classify digits from the MNIST dataset. The neural network is built from scratch using the Eigen library for matrix operations. The implementation includes the feed-forward network with ReLU and softmax activation functions, back-propagation for learning, and utilities to handle the MNIST data format.

The main motivation for this was as an exercise for my advanced machine-learning course.

## Features

- Neural network implementation with feed-forward, back-propagation, and training steps.
- Usage of the Eigen library for efficient matrix and vector operations.
- Handling MNIST dataset for both training and validation.
- Illustration of basic neural network concepts such as activation functions, loss computation, and gradient descent.

## Dependencies

- [Eigen Library](http://eigen.tuxfamily.org/): A C++ template library for linear algebra.

## Building and Running

1. **Clone the Repository:**

`git clone https://github.com/JakobSachs/simple_mnist_cpp.git && cd mnist-neural-network-cpp`

2. **Compile**:
If you have the requirements installed correctly, building the program should be doable by just simply running the Makefile

`make clean && make`

3. **Run the Program:**
Simply run the program:

`./main`

## Data Preparation

- Download the MNIST dataset from [MNIST Database](http://yann.lecun.com/exdb/mnist/).
- Place the downloaded files in a directory accessible to the program.

## Sample Output

```
epoch: 0, batch: 0/937, loss: 2.302
...
accuracy: 0.975
```


## License

This project is open-source and available under the [MIT License](LICENSE).

## Contact

For any queries or suggestions, feel free to reach out to jakobsachs1999@gmail.com.

