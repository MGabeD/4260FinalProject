import gym
env = gym.make('BipedalWalker-v3')
env.reset()

for _ in range(10000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    
    if done:
        env.reset()
env.close()
