# NN For ASL to English

This is an application built using [Digital Ocean's](https://www.digitalocean.com/community/tutorials/how-to-build-a-neural-network-to-translate-sign-language-into-english) tutorial as a reference.

The aim is to create a python application making use of PyTorch, OpenCV and onnx, to translate American Sign Language to English.

This will (hopefully) give me the experience needed in order to build my own converter to convert Barbadian Sign Language into English.

## Overview of steps

1. Preprocess data by applying one-hot encoding to labels and wrap data in Tensors.
2. Specify and training the model, by setting up a neural network using Pytorch, alongside setting hyper parameters.
3. Run predictions using the model, by evaluating the neural network on the validation data, then export model to ONNX.

## Note

I have the gpu version of torch and torch vision, and those are the ones inside of the `requirements_win_gpu.txt` file.
Otherwise, you can just install from the `requirements_win.txt`.