# Physics-Informed Neural Network Tutorial

## Overview

This tutorial introduces the concept of Physics-Informed Neural Networks (PINNs) and demonstrates how integrating the physics of a system into a neural network can enhance its performance. By the end of this tutorial, you will understand the basics of PINNs and be able to implement them for solving physics-based problems.


## Introduction

Physics-Informed Neural Networks (PINNs) combine the power of neural networks with the underlying physics equations governing a system. By incorporating known physical laws into the training process, PINNs can provide accurate predictions even with limited data.

## Prerequisites

Before starting the tutorial, ensure you have the following installed:

- Python (version 3.x)
- Optax
- NumPy
- Matplotlib

## Installation
### Method 1:
Clone this repository to your local machine:

``` git clone https://github.com/Mahsarnzh/Physics_Informed_Neural_Network.git ```

upload data.npz and PINNs_Temp.ipynb files on google cloud and run all

## Or

### Method 2:
Clone this repository to your local machine:

```bash
git clone https://github.com/Mahsarnzh/Physics_Informed_Neural_Network.git
cd Physics_Informed_Neural_Network
python -m venv venv
. ./venv/bin/activate
pip install optax 
python PINNS_Temp.py
