import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# Define the physics-informed neural network model

class PINNModel(tf.keras.Model):
    def __init__(self):
        super(PINNModel, self).__init__()

        # Neural network for solution u(x, t)
        self.u_model = self.build_model()

        # Neural network for residual equation f(x, t)
        self.f_model = self.build_model()

    def build_model(self):
        model = keras.Sequential([
            layers.InputLayer(input_shape=(2,)),  # x and t are inputs
            layers.Dense(50, activation='tanh', trainable=True),
            layers.Dense(50, activation='tanh', trainable=True),
            layers.Dense(1)
        ])
        # model.add(layers.Dense(units=50, input_dim=(2, 1), name='dense_5', trainable=True))
        return model

    def call(self, x, training=False):
        # Make sure to use both models in the call method
        u = self.u_model(x)
        f = self.f_model(x)
        return u, f


def pinn_loss(model, x, t):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    t = tf.convert_to_tensor(t, dtype=tf.float32)

    # Explicitly watch the time variable
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(t)

        # Concatenate inputs for both models
        inputs = tf.concat([x, t], axis=1)

        # Make sure to use both models in the call method
        u, f = model(inputs)

        # Governing PDE (e.g., heat conduction equation)
        pde = u * t - 0.01 * tf.math.pow(tf.math.sin(np.pi * x), 2)

    u_x = tape.gradient(u, x)
    u_t = tape.gradient(u, t)

    f_x = tape.gradient(f, x)
    f_t = tape.gradient(f, t)

    loss = tf.reduce_mean(tf.square(u_t - 0.01 * u_x - f_x)) + tf.reduce_mean(tf.square(f_t - pde))

    return loss


# Generate synthetic training data
def generate_data():
    x = np.random.uniform(low=-1, high=1, size=(100, 1))
    t = np.random.uniform(low=0, high=1, size=(100, 1))
    return x, t


# Main training loop
def train(model, optimizer, epochs):
    loss_ = []
    for epoch in range(epochs):
        x, t = generate_data()

        with tf.GradientTape() as tape:
            loss = pinn_loss(model, x, t)
            loss_.append(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # if epoch % 100 == 0:
        #     print(f'Epoch {epoch}, Loss: {loss.numpy()}')
    loss_ = np.array(loss_)
    # print(f'loss is:{loss_}')
    return loss_


# Create and train the PINN model
pinn_model = PINNModel()

# Example Adam optimizer with a custom learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# pinn_model.compile(optimizer=optimizer, loss='your_loss_function')

# Train the model using the train function
loss_ = train(pinn_model, optimizer=optimizer, epochs=1000)
plt.plot(loss_, label='Loss')
plt.xlabel('epoch')
plt.ylabel('Loss over each epoch')
plt.show()
