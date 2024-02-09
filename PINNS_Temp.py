import jax
import optax
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, grad
import matplotlib.pyplot as plt

key = jax.random.PRNGKey(2)

data = jnp.load('data.npz')
X1, X2, T, I = data['X1'], data['X2'], data['T'], data['I']

L = 1
k = 0.1
N_test  = 75
N_Model = 100
sigma = 2

I_train, I_test = I[N_test:], I[:N_test]

X_train = jnp.hstack((X1.reshape(-1,1)[I_train], X2.reshape(-1,1)[I_train]))
Y_train = T.ravel()[I_train] + sigma*jax.random.normal(key, shape=(len(I_train),))

X_test = jnp.hstack((X1.reshape(-1,1)[I_test], X2.reshape(-1,1)[I_test]))
Y_test = T.ravel()[I_test]

X_model = L*jax.random.uniform(key, shape=(N_Model,2))

q = lambda X: 250*jnp.exp(-20*(X[:,0]-L/6)**2-20*(X[:,1]-L/2)**2)



def Init(units, key):
    params = []
    keys = jax.random.split(key, num=4)

    w1 = jax.random.uniform(keys[0], (2, units[0]), minval=-1.0, maxval=1.0)
    b1 = jax.random.uniform(keys[1], (units[0],),   minval=-1.0, maxval=1.0)
    params.append([w1,b1])

    w2 = jax.random.uniform(keys[2], (units[0], units[1]), minval=-1.0, maxval=1.0)
    b2 = jax.random.uniform(keys[3], (units[1],), minval=-1.0, maxval=1.0)

    params.append([w2,b2])
    return params


def Forward_NN_Method_1(params, x1, x2):

    input = jnp.hstack((x1.reshape(-1,1), x2.reshape(-1,1)))
    l1 = jnp.tanh(input @ params[0][0] + params[0][1])
    l2 = l1 @ params[1][0] + params[1][1]
    return l2.squeeze()


units, learning_rate, epochs = [40,1], 1e-4, 100000
params = Init(units, key)


@jit
def Loss_NN_Method_1(params, X_data, Y_data):

    loss = (1/N_test)*(jnp.mean((Y_train - Forward_NN_Method_1(params, X_train[:,0], X_train[:,1] ))**2))

    return loss



Gradients = grad(Loss_NN_Method_1)

params = Init(units, key)

optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

# The function "step" performs one step of the optimization - it updates the weights based on the gradient and it should be called in the optimization iterations. You do not need to change this function in this project.
# The function uses "adams", which is a gradient-based optimiation algorithm. It performs better than the vanilla gradient descent method.
# "params" is a list that contains all regression parameters.
# "opt_state" contains a set of optimization parameters that the algorithm uses. You do not need to use opt_state anywhere else.
# This is just used in the optimization iterations.
@jit
def step(params, opt_state):
    gradients = Gradients(params, X_train, Y_train)
    updates, opt_state = optimizer.update(gradients, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state


MSE_Loss_Test_Data_NN_Method_1 = []
MSE_Loss_Train_Data_NN_Method_1 = []
for i in range(epochs):

    params, opt_state = step(params, opt_state)

    MSE_Loss_Test_Data_NN_Method_1.append(Loss_NN_Method_1(params, X_test, Y_test))
    MSE_Loss_Train_Data_NN_Method_1.append(Loss_NN_Method_1(params, X_train, Y_train))

fig, ax = plt.subplots(figsize=(10, 10))

x__ = jnp.linspace(0, epochs, epochs)
ax.plot(x__, jnp.array(MSE_Loss_Train_Data_NN_Method_1), c='green')
ax.plot(x__, jnp.array(MSE_Loss_Test_Data_NN_Method_1), c='blue')
ax.set_xlabel('iteration')
ax.set_ylabel('MSE_Loss')
plt.title('_Regression_Method_1')
plt.legend([
  'MSE_NN_Method_1_Loss_Train',
  'MSE_NN_Method_1_Loss_Test',
])
plt.grid(True)
plt.show()

Forward_NN_Method_1 = Forward_NN_Method_1(params, X1, X2)
Forward_NN_Method_1.shape
Forward_NN_ = Forward_NN_Method_1.reshape(101,101)
fig = plt.figure(figsize=(10,10))
plt.contourf(X1, X2, Forward_NN_, cmap='jet', levels = 20)
plt.colorbar()

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('_NN_Method_1')

def Init(units, key):
    params = []
    keys = jax.random.split(key, num=4)

    w1 = jax.random.uniform(keys[0], (2, units[0]), minval=-1.0, maxval=1.0)
    b1 = jax.random.uniform(keys[1], (units[0],),   minval=-1.0, maxval=1.0)
    params.append([w1,b1])

    w2 = jax.random.uniform(keys[2], (units[0], units[1]), minval=-1.0, maxval=1.0)
    b2 = jax.random.uniform(keys[3], (units[1],), minval=-1.0, maxval=1.0)

    params.append([w2,b2])
    return params


def Forward_PINNS_Regression(params, x1, x2):

    input = jnp.hstack((x1.reshape(-1,1), x2.reshape(-1,1)))
    l1 = jnp.tanh(input @ params[0][0] + params[0][1])
    l2 = l1 @ params[1][0] + params[1][1]
    return l2.squeeze()


units, learning_rate, epochs = [40,1], 1e-4, 100000
params = Init(units, key)

d2y_dx1 = jit(vmap(grad(grad(Forward_PINNS_Regression, 1), 1), in_axes=(None, 0, 0)))
d2y_dx2 = jit(vmap(grad(grad(Forward_PINNS_Regression, 2), 2), in_axes=(None, 0, 0)))
k = 0.1


def R_PINNS_Regression(params, X):
    R = (k)*(d2y_dx1(params, X[:,0], X[:,1]) + d2y_dx2(params, X[:,0], X[:,1])) + q(X)
    return R


@jit
def Loss_PINNS_Regression(params, X_data, Y_data, X):

    # loss = jax.numpy.linalg.norm((Y_data - Forward(params, X_data[:,0], X_data[:,1] ))) + (1/N_Model)*(jax.numpy.linalg.norm(R(params, X)))
    loss = (1/N_Model)*(jnp.mean((R_PINNS_Regression(params, X_model))**2)) + (1/N_test)*(jnp.mean((Y_train - Forward_PINNS_Regression(params, X_train[:,0], X_train[:,1] ))**2))
    return loss



Gradients = grad(Loss_PINNS_Regression)

params = Init(units, key)



optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

# The function "step" performs one step of the optimization - it updates the weights based on the gradient and it should be called in the optimization iterations. You do not need to change this function in this project.
# The function uses "adams", which is a gradient-based optimiation algorithm. It performs better than the vanilla gradient descent method.
# "params" is a list that contains all regression parameters.
# "opt_state" contains a set of optimization parameters that the algorithm uses. You do not need to use opt_state anywhere else.
# This is just used in the optimization iterations.
@jit
def step(params, opt_state):
    gradients = Gradients(params, X_train, Y_train, X_model)
    updates, opt_state = optimizer.update(gradients, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state


MSE_Loss_Test_Data_PINNS = []
MSE_Loss_Train_Data_PINNs = []
for i in range(epochs):

    params, opt_state = step(params, opt_state)

    MSE_Loss_Test_Data_PINNS.append(Loss_PINNS_Regression(params, X_test, Y_test, X_model))
    MSE_Loss_Train_Data_PINNs.append(Loss_PINNS_Regression(params, X_train, Y_train, X_model))


# Forward(params, X_train,  X_train)

fig, ax = plt.subplots(figsize=(10, 10))

x__ = jnp.linspace(0, epochs, epochs)
# ax.plot(x__, jnp.array(MSE_Loss_Train_Data_PINNs), c='green')
ax.plot(x__, jnp.array(MSE_Loss_Test_Data_PINNS), c='blue')
ax.set_xlabel('iteration')
ax.set_ylabel('MSE_Loss')
plt.title('Part 2, MSE_PINNs')
plt.legend([
  'MSE_Loss_Train_Data_PINNs',
  'MSE_Loss_Test_Data_PINNS',
])
plt.grid(True)
plt.show()

testx = Forward_PINNS_Regression(params, X1, X2)
testx.shape
plotx = testx.reshape(101,101)
fig = plt.figure(figsize=(10,10))
plt.contourf(X1, X2, plotx, cmap='jet', levels = 20)
plt.colorbar()

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('_PINNS_Method_1')

def Init(units, key):
    params = []
    keys = jax.random.split(key, num=5)

    w1 = jax.random.uniform(keys[0], (2, units[0]), minval=-1.0, maxval=1.0)
    b1 = jax.random.uniform(keys[1], (units[0],),   minval=-1.0, maxval=1.0)

    params.append([w1,b1])

    w2 = jax.random.uniform(keys[2], (units[0], units[1]), minval=-1.0, maxval=1.0)
    b2 = jax.random.uniform(keys[3], (units[1],), minval=-1.0, maxval=1.0)
    k = jax.random.uniform(keys[4], (1,), minval=0.5, maxval=0.5)

    params.append([w2,b2,k])
    return params


def Forward(params, x1, x2):

    input = jnp.hstack((x1.reshape(-1,1), x2.reshape(-1,1)))
    l1 = jnp.tanh(input @ params[0][0] + params[0][1])
    l2 = l1 @ params[1][0] + params[1][1]
    return l2.squeeze()


units, learning_rate, epochs = [40,1], 1e-4, 100000
params = Init(units, key)

params[1][2]

d2y_dx1 = jit(vmap(grad(grad(Forward, 1), 1), in_axes=(None, 0, 0)))
d2y_dx2 = jit(vmap(grad(grad(Forward, 2), 2), in_axes=(None, 0, 0)))


def R(params, X):
    R = (params[1][2])*(d2y_dx1(params, X[:,0], X[:,1]) + d2y_dx2(params, X[:,0], X[:,1])) + q(X)
    return R


@jit
def Loss(params, X_data, Y_data, X):

    loss = (1/N_Model)*(jnp.mean((R(params, X_model))**2)) + (1/N_test)*(jnp.mean((Y_train - Forward(params, X_train[:,0], X_train[:,1]))**2))
    return loss

Gradients = grad(Loss)

params = Init(units, key)

optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)


@jit
def step(params, opt_state):
    gradients = Gradients(params, X_train, Y_train, X_model)
    updates, opt_state = optimizer.update(gradients, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state

MSE_Loss_Test_Data_PINNS = []
MSE_Loss_Train_Data_PINNs = []
Regressed_K = []
for i in range(epochs):

    params, opt_state = step(params, opt_state)

    Regressed_K.append(params[1][2])

    MSE_Loss_Test_Data_PINNS.append(Loss(params, X_test, Y_test, X_model))
    MSE_Loss_Train_Data_PINNs.append(Loss(params, X_train, Y_train, X_model))

Regressed_K_ = params[1][2]
Regressed_K_

fig, ax = plt.subplots(figsize=(10, 10))

x__ = jnp.linspace(0, epochs, epochs)
ax.plot(x__, jnp.array(Regressed_K), c='green')
ax.set_xlabel('iteration')
ax.set_ylabel('K')
plt.title('Part 7, k_UNKNOWN')

fig, ax = plt.subplots(figsize=(10, 10))

x__ = jnp.linspace(0, epochs, epochs)
ax.plot(x__, jnp.array(MSE_Loss_Train_Data_PINNs), c='green')
ax.plot(x__, jnp.array(MSE_Loss_Test_Data_PINNS), c='blue')

ax.set_xlabel('iteration')
ax.set_ylabel('MSE_Loss')
plt.title('Part 7, MSE_PINNs_k_UNKNOWN')
plt.legend([
  'MSE_Loss_Train_Data_PINNs',
  'MSE_Loss_Test_Data_PINNS',
])
plt.grid(True)
plt.show()

Forward_ = Forward(params, X1, X2)
Forward_.shape
Forward__ = Forward_.reshape(101,101)
fig = plt.figure(figsize=(10,10))
plt.contourf(X1, X2, Forward__, cmap='jet', levels = 20)
plt.colorbar()
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('_PINNS_Method_1_Part_7')


import jax
import optax
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, grad
import matplotlib.pyplot as plt

key = jax.random.PRNGKey(2)
data = jnp.load('data.npz')

X1, X2, T, I = data['X1'], data['X2'], data['T'], data['I']

K  = 25*25
L = 1
k = 0.1
N_test  = 75
N_Model = 500
sigma = 2

I_train, I_test = I[N_test:], I[:N_test]

X_train = jnp.hstack((X1.reshape(-1,1)[I_train], X2.reshape(-1,1)[I_train]))
Y_train = T.ravel()[I_train] + sigma*jax.random.normal(key, shape=(len(I_train),))

X_test = jnp.hstack((X1.reshape(-1,1)[I_test], X2.reshape(-1,1)[I_test]))
Y_test = T.ravel()[I_test]

X_model = L*jax.random.uniform(key, shape=(N_Model,2))

q = lambda X: 250*jnp.exp(-20*(X[:,0]-L/6)**2-20*(X[:,1]-L/2)**2)

M = 25
K  = 25*25
s  = 0.25
mu = jax.random.uniform(key, shape=(2, K))

def Init(K, key):
    return [jax.random.uniform(key, (K,1), minval=-1.0, maxval=1.0)]


def Forward(params, x1, x2):
  phi = (jnp.exp(-(x1.reshape(-1,1)- mu[0][:])**2/ s**2))*(jnp.exp(-(x2.reshape(-1,1)- mu[1][:])**2/ s**2))
  l1 = phi @ params[0]
  return l1.squeeze()

learning_rate, epochs = 1e-1, 100000
params_ = Init(K, key)


@jit
def Loss(params, X_data, Y_data):

    loss = (1/N_test)*(jnp.mean((Y_data - Forward(params, X_data[:,0], X_data[:,1] ))**2))
    return loss

Gradients = grad(Loss)


optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params_)

@jit
def step(params_, opt_state):
    gradients = Gradients(params_, X_train, Y_train)
    updates, opt_state = optimizer.update(gradients, opt_state, params_)
    params_ = optax.apply_updates(params_, updates)
    return params_, opt_state

MSE_Loss_Test_Data_Regression = []
MSE_Loss_Train_Data_Regression = []
for i in range(epochs):

    params_, opt_state = step(params_, opt_state)

    MSE_Loss_Test_Data_Regression.append(Loss(params_, X_test, Y_test))
    MSE_Loss_Train_Data_Regression.append(Loss(params_, X_train, Y_train))

fig, ax = plt.subplots(figsize=(10, 10))

x__ = jnp.linspace(0, epochs, epochs)
ax.plot(x__, jnp.array(MSE_Loss_Test_Data_Regression), c='green')
ax.plot(x__, jnp.array(MSE_Loss_Train_Data_Regression), c='blue')
ax.set_xlabel('iteration')
ax.set_ylabel('MSE_Loss')
plt.title('Method2_ MSE_Regression')
plt.legend([
  'MSE_Loss_Train_Data_Regression',
  'MSE_Loss_Test_Data_Regression',
])
plt.grid(True)
plt.show()

testx = Forward(params_, X1, X2)
testx.shape
plotx = testx.reshape(101,101)
fig = plt.figure(figsize=(10,10))
plt.contourf(X1, X2, plotx, cmap='jet', levels = 20)
plt.colorbar()

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('_Regression_Method_2')

M = 25
K  = 25*25
s  = 0.25
mu = jax.random.uniform(key, shape=(2, K))

def Init(K, key):
    return [jax.random.uniform(key, (K,1), minval=-1.0, maxval=1.0)]


def Forward(params, x1, x2):
  phi = (jnp.exp(-(x1.reshape(-1,1)- mu[0][:])**2/ s**2))*(jnp.exp(-(x2.reshape(-1,1)- mu[1][:])**2/ s**2))
  l1 = phi @ params[0]
  return l1.squeeze()

learning_rate, epochs = 1e-1, 100000
params_ = Init(K, key)

d2y_dx1 = jit(vmap(grad(grad(Forward, 1), 1), in_axes=(None, 0, 0)))
d2y_dx2 = jit(vmap(grad(grad(Forward, 2), 2), in_axes=(None, 0, 0)))

def R(params, X):
    R = (k)*(d2y_dx1(params, X[:,0], X[:,1]) + d2y_dx2(params, X[:,0], X[:,1])) + q(X)
    return R


@jit
def Loss(params, X_data, Y_data, X):

    loss = (1/N_Model)*(jnp.mean((R(params, X))**2)) + (1/N_test)*(jnp.mean((Y_data - Forward(params, X_data[:,0], X_data[:,1] ))**2))
    return loss

Gradients = grad(Loss)



optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params_)

@jit
def step(params_, opt_state):
    gradients = Gradients(params_, X_train, Y_train, X_model)
    updates, opt_state = optimizer.update(gradients, opt_state, params_)
    params_ = optax.apply_updates(params_, updates)
    return params_, opt_state

MSE_Loss_Test_Data_PIR = []
MSE_Loss_Train_Data_PIR = []
for i in range(epochs):

    params_, opt_state = step(params_, opt_state)

    MSE_Loss_Test_Data_PIR.append(Loss(params_, X_test, Y_test, X_model))
    MSE_Loss_Train_Data_PIR.append(Loss(params_, X_train, Y_train, X_model))

fig, ax = plt.subplots(figsize=(10, 10))

x__ = jnp.linspace(0, epochs, epochs)
ax.plot(x__, jnp.array(MSE_Loss_Test_Data_PIR), c='green')
ax.plot(x__, jnp.array(MSE_Loss_Train_Data_PIR), c='blue')
ax.set_xlabel('iteration')
ax.set_ylabel('MSE_Loss')
plt.title('Method 2, MSE_PIR')
plt.legend([
  'MSE_Loss_Train_Data_PIR',
  'MSE_Loss_Test_Data_PIR',
])
plt.grid(True)
plt.show()

testx = Forward(params_, X1, X2)
testx.shape
plotx = testx.reshape(101,101)
fig = plt.figure(figsize=(10,10))
plt.contourf(X1, X2, plotx, cmap='jet', levels = 20)
plt.colorbar()

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('_Physics_Informed_Regression_Method_2')
