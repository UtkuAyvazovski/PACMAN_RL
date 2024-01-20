from model import *
import os
import numpy as np

class RL_class():

    def __init__(self):
        self.discount_factor=0.9
        self.model=KI_CNN_model()
        self.target_model=KI_CNN_model()
        self.update_target_model(tau=0)

        self.database_path="data/training_data.npz"
        
        self.utilize_database=False
        if self.utilize_database:
            self.database_exist=False
        else:
            self.database_exist= os.path.isfile(self.database_path)

        self.criterion = nn.MSELoss()

        
        
        self.batch_size=100
        self.epsilon=0.3

        self.model_lr=0.0001
        self.optimizer=optim.Adam(self.model.parameters(), lr=self.model_lr)
    
    def greedy_action(self, states):
        """
        Take the action the model sees as the best
        """
        with torch.no_grad():
            _, action_index = torch.max(self.model(torch.from_numpy(states).float()), 1)
        return action_index

    def greedy_epsilon_action(self, states):
        """
        Take random actions according to the epsilone
        """
        if np.random.rand()>0.3:
            return self.greedy_action(states)
        else:
            return torch.from_numpy(np.array([np.random.randint(5)]))

    def value_func(self, states):
        """
        Computes the value function of the model
        Parameters:
        -obs: observation of the agent 
        """
        return self.model(states)

    def target_value_func(self, reward, dones, next_state):
        """
        Computes the target value function
        Parameters:
        -next_obs: observation of the agent of the next state
        Return value:
        -target value function
        """
        return reward[:,0]#+(dones[:,0]==0)*torch.max(self.target_model(next_state), axis=1)[0]

    def train_model(self):
        """
        Train the original model
        """
        states, actions, rewards, dones, next_states=self.get_training_batch()
        for i in range(10):
            self.optimizer.zero_grad()
            outputs = self.value_func(states)[torch.arange(self.batch_size), actions[:,0].long()]
            target_outputs = self.target_value_func(rewards, dones, next_states)
            loss = self.criterion(outputs, target_outputs.detach())
            print(loss)
            loss.backward()
            self.optimizer.step()

    def update_target_model(self, tau=0.001):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def get_training_batch(self):
        """
        Return random training data equal to the batch size
        """
        if self.utilize_database:
            data = np.load(self.database_path)
            states=data["states"]
            num_samples=states.shape[0]
            random_batch=np.random.randint(num_samples, size=self.batch_size)
            
            states = states[random_batch]
            actions=data["actions"][random_batch]
            rewards=data["rewards"][random_batch]
            dones=data["dones"][random_batch]
            next_states=data["next_states"][random_batch]
        else:
            num_samples=self.states.shape[0]
            random_batch=np.random.randint(num_samples, size=self.batch_size)
            states=self.states[random_batch]
            actions=self.actions[random_batch]
            rewards=self.rewards[random_batch]
            dones=self.dones[random_batch]
            next_states=self.next_states[random_batch]


        states=torch.from_numpy(states).float()
        actions=torch.from_numpy(actions).float()
        rewards=torch.from_numpy(rewards).float()
        dones=torch.from_numpy(dones).float()
        next_states=torch.from_numpy(next_states).float()

        return states, actions, rewards, dones, next_states

    def insert_data(self, state, action, reward, done, next_state):
        """
        Insert the new data into the database
        Parameters:
        -state: state of the agent
        -action: action of the agent
        -reward: reward of the agent
        -done: terminal state reached
        -next_state: the next state of the agent(if None then the env was reset) 
        """
        #insert the data into the database
        saveable_state=np.expand_dims(state, axis=0)
        saveable_action=np.expand_dims(action, axis=0)
        saveable_reward=np.expand_dims(np.array([reward]), axis=0)
        saveable_done=np.expand_dims(np.array([done]), axis=0)
        saveable_next_state=np.expand_dims(next_state, axis=0)
        
        if not self.database_exist:
            if self.utilize_database:
                np.savez(self.database_path, 
                states=saveable_state, 
                actions=saveable_action, 
                rewards=saveable_reward, 
                dones=saveable_done, 
                next_states=saveable_next_state)
            else:
                self.states=saveable_state
                self.actions=saveable_action
                self.rewards=saveable_reward
                self.dones=saveable_done
                self.next_states=saveable_next_state
                self.database_exist=True

        else:
            if self.utilize_database:
                data = np.load(self.database_path)
                np.savez(self.database_path,
                states=np.concatenate((data["states"], saveable_state), axis=0),
                actions=np.concatenate((data["actions"], saveable_action), axis=0),
                rewards=np.concatenate((data["rewards"], saveable_reward), axis=0),
                dones=np.concatenate((data["dones"], saveable_done), axis=0),
                next_states=np.concatenate((data["next_states"], saveable_next_state), axis=0))
            else:
                
                self.states=np.concatenate((self.states, saveable_state), axis=0)
                self.actions=np.concatenate((self.actions, saveable_action), axis=0)
                self.rewards=np.concatenate((self.rewards, saveable_reward), axis=0)
                self.dones=np.concatenate((self.dones, saveable_done), axis=0)
                self.next_states=np.concatenate((self.next_states, saveable_next_state), axis=0)

