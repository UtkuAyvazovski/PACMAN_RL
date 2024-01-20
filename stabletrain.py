import gym
import cv2
from matplotlib import pyplot as plt
from model import *
from stable_baselines3.common.atari_wrappers import *
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import VecFrameStack

class CustomPreprocessEnv(gym.Wrapper):
    def __init__(self, env, preprocess_fn):
        super(CustomPreprocessEnv, self).__init__(env)
        self.preprocess_fn = preprocess_fn

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.preprocess(observation)

    def step(self, action):
        observation, reward, done, info, extra = self.env.step(action)
        return self.preprocess(observation), reward, done, info, extra

    def preprocess(self, observation):
        # Apply your custom preprocessing logic here
        preprocessed_observation = self.preprocess_fn(observation)
        return preprocessed_observation

def custom_preprocessing(observation):
    """
    The observation has unnecessary information
    """
    if type(observation)==tuple:
        return (preprocessing(observation[0]), observation[1])
    else:
        return preprocessing(observation)

def preprocessing(observation):
    #Take necessary part of the image
    current_observation = observation[0:173,:,:]
    #Make image smaller
    #current_observation = cv2.resize(current_observation, dsize=(current_observation.shape[0]//2, current_observation.shape[1]//2), interpolation=cv2.INTER_LINEAR)
    #Grayscale the image
    #current_observation = np.expand_dims(np.dot(current_observation, [0.299, 0.587, 0.114]), axis=0)
    
    return current_observation
    

# Define the MsPacman environment
env = gym.make('ALE/MsPacman-v5')
env.action_space=gym.spaces.Discrete(5)

#print(env.observation_space)
#input("check")
preprocessed_env = CustomPreprocessEnv(env, custom_preprocessing)
state_shape=preprocessed_env.reset()[0].shape
env.observation_space = gym.spaces.Box(low=0, high=255, shape=(state_shape[0], state_shape[1], state_shape[2]), dtype=np.uint8)
env = CustomPreprocessEnv(env, custom_preprocessing)
# Define the DQN model
model = DQN("CnnPolicy", env, 
    verbose=1, 
    buffer_size=15000, 
    learning_starts=1000,
    device="auto", 
    train_freq=(100, "step"),
    exploration_final_eps=0.3,
    gradient_steps=10,
    batch_size=100,
    gamma=0.8
    )
model.learn(total_timesteps=1000000)

# Save the trained model
model.save("dqn_mspacman")

# Main loop to interact with the environment
total_reward=0
obs = env.reset()[0]
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    total_reward+=reward
    if done:
        break;

# Close the environment
env.close()
print(total_reward)
#Use total reward as performance metric
print("worked")
