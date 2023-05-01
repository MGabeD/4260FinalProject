import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import gym
import Box2D
import random
from deap import base, creator, tools, algorithms
import pickle

env = gym.make('BipedalWalker-v3')

# This will help since it makes the game not stop after a specific amount of steps
env = env.env 
env.reset()

# measurements of the env (total) - number of input dimensions
in_dim = env.observation_space.shape[0]
# possible actions - number of output dimensions
out_dim = env.action_space.shape[0]

# model_weights_as_vector and model_weights_as_matrix are directly taken from 
# tutorial for another model (cartpole)
def model_weights_as_vector(model):
    weights_vector = []

    for layer in model.layers: 
        if layer.trainable:
            layer_weights = layer.get_weights()
            for l_weights in layer_weights:
                vector = np.reshape(l_weights, newshape=(l_weights.size))
                weights_vector.extend(vector)

    return np.array(weights_vector)

def model_weights_as_matrix(model, weights_vector):
    weights_matrix = []

    start = 0
    for layer_idx, layer in enumerate(model.layers): 
        layer_weights = layer.get_weights()
        if layer.trainable:
            for l_weights in layer_weights:
                layer_weights_shape = l_weights.shape
                layer_weights_size = l_weights.size
        
                layer_weights_vector = weights_vector[start:start + layer_weights_size]
                layer_weights_matrix = np.reshape(layer_weights_vector, newshape=(layer_weights_shape))
                weights_matrix.append(layer_weights_matrix)
        
                start = start + layer_weights_size
        else:
            for l_weights in layer_weights:
                weights_matrix.append(l_weights)

    return weights_matrix

def model_build(in_dim=in_dim, out_dim=out_dim):
    # Defining the model with keras
    model = Sequential()
    model.add(Dense(64, input_dim=in_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(out_dim, activation='tanh'))

    # Compile the model with mean squared error loss and Adam optimizer
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    
    return model

def evaluate(individual,award=0):
    env.reset()
    obs1 = env.reset()
    model = model_build()
    model.set_weights(model_weights_as_matrix(model, individual))
    done = False
    step = 0
    while (done == False) and (step<=1000):
        obs2 = np.expand_dims(obs1, axis=0)
        obs3 = []
        for i in range(in_dim):  
            obs3.append(obs2[0][i])
        obs4 = np.array(obs3).reshape(-1)
        obs = np.expand_dims(obs4, axis=0)
        selected_move1 = model.predict(obs)
        obs2, reward, done, info = env.step(selected_move1[0])
        award += reward
        step = step+1
        obs1 = obs2
    return (award,)


model = model_build()
ind_size = model.count_params()
print(ind_size)
print(model.summary())

creator.create("Max", base.Fitness, weights=(1.0,))
creator.create("Indiv", list, fitness=creator.Max)
toolbox = base.Toolbox()
toolbox.register("weight_bin", np.random.uniform,-1,1)
toolbox.register("indiv", tools.initRepeat, creator.Indiv, toolbox.weight_bin, n=ind_size)
toolbox.register("population", tools.initRepeat, list, toolbox.indiv)


toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)



stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("Mean", np.mean)
stats.register("Max", np.max)
stats.register("Min", np.min)



pop = toolbox.population(n=100)
hof = tools.HallOfFame(1)


pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.2, ngen=30, halloffame=hof, stats=stats)


with open("bipedalWalker.pkl", "wb") as cp_file:
    pickle.dump(hof.items[0], cp_file)

