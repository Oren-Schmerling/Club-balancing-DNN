import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.losses import Huber
from keras.metrics import mean_squared_error

keras.utils.disable_interactive_logging()

class ClubAgent:
    def __init__(self, state_size, output_size, episodeLength, numHiddenLayers, LRInitial, LRDecay):
        self.num_actions = output_size
        self.n_actions = self.num_actions
        #some hyperparameters:
        #
        # lr - learning rate
        # gamma - discount factor
        # exploration_proba - initial exploration probability
        # exploration_proba_decay - decay of exploration probability
        # batch_size - size of experiences we sample to train the DNN
        self.gamma = 0.99
        self.exploration_proba = 1
        self.exploration_proba_decay = 0.03
        self.choiceCutoff = .6
        self.batch_size = int(episodeLength/10)
        self.memory_buffer = []
        self.memory_buffer_reward = []
        self.max_memory_buffer = episodeLength
        self.lr_schedule = keras.optimizers.schedules.ExponentialDecay(LRInitial,decay_steps=self.batch_size,decay_rate=LRDecay)


        #create model having two hidden layers of 12 neurons
        #the first layer has the same size as state size
        #the last layer has the size of the action space
        self.model = Sequential([
            Dense(units=5, input_dim = state_size, activation = 'relu'),
            # Dense(units=4, activation = 'relu'),
            # Dense(units=3, activation = 'relu'),
            # Dense(units=4, activation = 'relu'),
            # Dense(units=self.n_actions, activation = 'linear')
        ])
        for i in range(numHiddenLayers):
            self.model.add(Dense(units=6, activation = 'relu'))
        self.model.add(Dense(units=self.n_actions, activation = 'linear'))
        self.model.compile(loss = Huber(), optimizer = Adam(learning_rate = self.lr_schedule))

    def getProb(self):
        return self.exploration_proba

    #the agent computes the action to perform given a state
    def compute_action(self, current_state):
        choices = self.model.predict(current_state)[0]
        #list of 4 bits, representing pressing w,a,s,d in that order
        actionList = []
        for i in range(self.num_actions):
            if np.random.uniform(0,1) < self.exploration_proba:
                actionList.append(np.random.choice(range(2)))
            else:
                if choices[i] > self.choiceCutoff:
                    actionList.append(1)
                else:
                    actionList.append(0)
        return actionList

    
    #when an episode is finished, we update the exploration proba using epsilon greedy algorithm
    def update_exploration_probability(self):
        self.exploration_proba = self.exploration_proba * np.exp(-self.exploration_proba_decay)

    #at each step, we store the corresponding experience
    def store_episode(self, current_state, action, reward, next_state):
        #we use a dictionary to store them
        self.memory_buffer.append({
            "current_state":current_state,
            "action":action,
            "reward":reward,
            "next_state":next_state,
        })
        if len(self.memory_buffer) > self.max_memory_buffer:
            self.memory_buffer.pop(0)
    
    def store_episode_reward(self, current_state, action, reward, next_state):
        #we use a dictionary to store them
        self.memory_buffer_reward.append({
            "current_state":current_state,
            "action":action,
            "reward":reward,
            "next_state":next_state,
        })

            
    def disableRandom(self):
        self.exploration_proba = 0
        self.exploration_proba_decay = 1

    def reduceRandom(self):
        self.exploration_proba = 0.8
        self.exploration_proba_decay = 0.07


    #at the end of each episode, we train the model
            # TODO: FIX THIS
    def train(self):
        batch_sample = []
        #select a batch of random experiences
        for i in range(self.batch_size):
            if (self.memory_buffer.__len__() < 1):
                break
            index = np.random.randint(0,len(self.memory_buffer))
            batch_sample.append(self.memory_buffer.pop(index))

        #add rewarded events to the memory buffer
        for sample in self.memory_buffer_reward:
            batch_sample.insert(0, sample)

        #we iterate over the selected experiences
        for experience in batch_sample:
            #we compute the Q-target using bellman optimality equation
            q_target = [0,0]
            reward = experience["reward"]
            q_target[0] = self.gamma * self.model.predict(experience["next_state"])[0][0]
            q_target[1] = self.gamma * self.model.predict(experience["next_state"])[0][1]
            for i in range(2):
                if experience["action"][i] == 1:
                    q_target[i] += reward
            #train the model
            self.model.fit(experience["current_state"], np.array([q_target]))
        
        self.memory_buffer.clear()
        self.memory_buffer_reward.clear()


        