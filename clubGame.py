import pygame
import math
from neural_network_agent import ClubAgent
import pathlib
import numpy as np

class Game:
    def __init__(self):
        
        #                               parameters
        self.gameSpeed = 1
        self.maxfps = 60
        self.printScreen = True
        self.playerRad = 24
        self.ballRad = 16
        self.playerSpeed = 600
        self.screenWidth = 1280
        self.screenHeight = 720
        self.grav = 20
        self.entropy = .999
        self.playerColor = "white"
        self.tetherLen = 256
        self.tetherWid = 12

        self.reset()

        #
        #                               link up DNN to game
        #
        self.trialName = "normalized"
        self.loading = False
        self.saveFrequency = 50
        episodeLength = 300
        self.variables = [self.playerPos[0], self.ballPos[0], self.ballPos[1]]
        state_size = 3
        self.max_iteration_ep = episodeLength
        self.rewardVal = 1
        self.rewardThreshhold = 8
        self.output_size = 2
        self.agent = ClubAgent(state_size, self.output_size, episodeLength)
        self.total_steps = 0
        self.numEpisodes = 0
        self.numsave = 0
        self.averageMemory = []
        self.memBuffer = 20


        pathlib.Path(self.trialName+"/").mkdir(exist_ok=True) 
        #
        # OPTIONAL: Load previous weights
        #
        
        if self.loading == True:
            self.numsave = 2
            checkpoint_path = "./normalized/checkpoint"+str(self.numsave)
            self.numEpisodes = self.numsave * self.saveFrequency
            self.agent.model.load_weights(checkpoint_path)
            # pick one or the other:
            self.agent.disableRandom()
            # self.agent.reduceRandom()
        else:
            self.numsave = 0
            #Saving the parameters
            with open(self.trialName+"/LOG.txt", "a+") as f:
                    f.write("Reward = "+str(self.rewardVal) + "\n")
                    f.write("Reward region = "+str(self.rewardThreshhold) + "\n")
                    f.write("Output size = "+str(self.output_size) + "\n")
                    f.write("Input size = "+str(state_size) + "\n")
                    f.write("learning rate = "+str(self.agent.lr)+ "\n")
                    f.write("gamma = "+str(self.agent.gamma)+"\n")
                    f.write("random chance = "+str(self.agent.exploration_proba)+"\n")
                    f.write("random decay = "+str(self.agent.exploration_proba_decay)+"\n")
                    f.write("batch size = "+str(self.agent.batch_size)+"\n")
                    f.write("choice cutoff = "+str(self.agent.choiceCutoff)+"\n")

        
        
        #                               pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((self.screenWidth, self.screenHeight))
        self.clock = pygame.time.Clock()
        self.running = True
        self.dt = 0
        self.ballSpeed = 0
        
    def reset(self):
        self.playerPos = [self.screenWidth/2, self.screenHeight/2]
        self.ballPos = [self.playerPos[0] + np.random.uniform(-10,10), self.playerPos[1]-self.tetherLen]
        self.prevPos = [self.ballPos[0], self.ballPos[1]]
        self.ballVel = [0.0, 0.0]
        self.mult = 0
        self.agentActions = [0,0,0,0]
        self.first = True
        self.reward = 0
        self.dt = .02

    def normalize(self, var):
        #[self.playerPos[0], self.ballPos[0], self.ballPos[1]]
        ret = [0,0,0]
        ret[0] = var[0]/self.screenWidth
        ret[1] = var[1]/self.screenWidth
        ret[2] = var[2]/self.screenHeight
        return ret
                    
    def simulation(self): #                       MAIN FUNCTION
        
        while True: #               looping episodes

            self.reset()
            current_state = 0
            self.numEpisodes += 1

            for frame in range(self.max_iteration_ep): #  game loop    

                for event in pygame.event.get(): # check if game has been quit
                    if (event.type ==pygame.QUIT):
                        self.running = False
                        pygame.quit()
                                    
                if self.first == False:
                    if (self.reward != 0):
                        self.variables = [self.playerPos[0], self.ballPos[0], self.ballPos[1]]
                        variables = self.normalize(self.variables)
                        next_state = np.array([variables])
                        self.agent.store_episode_reward(current_state, self.agentActions, self.reward, next_state)
                        self.reward = 0
                    else:
                        self.variables = [self.playerPos[0], self.ballPos[0], self.ballPos[1]]
                        variables = self.normalize(self.variables)
                        next_state = np.array([variables])
                        self.agent.store_episode(current_state, self.agentActions, self.reward, next_state)
                    
                self.first = False
                #                           decision making
                self.variables = [self.playerPos[0], self.ballPos[0], self.ballPos[1]]
                variables = self.normalize(self.variables)
                current_state = np.array([variables])
                if (self.ballPos[1] < (self.playerPos[1] - (self.tetherLen - self.rewardThreshhold))):
                    self.reward = self.rewardVal
                self.agentActions = self.agent.compute_action(current_state)
                        

                #                           controls
                moveX = 0
                moveY = 0
                keys = pygame.key.get_pressed()
                if keys[pygame.K_SPACE]:
                    if self.printScreen == False:
                        self.printScreen = True
                    else:
                        self.printScreen = False
                        self.screen.fill("black")
                
                #                                       for 4 directional control
                # if ((keys[pygame.K_w] or (self.agentActions[0] == 1)) and (self.playerPos[1] > 0)):
                #     moveY -= 1
                # if ((keys[pygame.K_a] or (self.agentActions[1] == 1)) and (self.playerPos[0] > 0)):
                #     moveX -= 1
                # if ((keys[pygame.K_s] or (self.agentActions[2] == 1)) and (self.playerPos[1] < self.screenHeight)):
                #     moveY += 1
                # if ((keys[pygame.K_d] or (self.agentActions[3] == 1)) and (self.playerPos[0] < self.screenWidth)):
                #     moveX += 1
                #                                       for 2 directional control
                if ((keys[pygame.K_a] or (self.agentActions[0] == 1)) and (self.playerPos[0] > 0)):
                    moveX -= 1
                if ((keys[pygame.K_d] or (self.agentActions[1] == 1)) and (self.playerPos[0] < self.screenWidth)):
                    moveX += 1
                
                    
                #player movement vector
                vectLen = ((moveX ** 2)+(moveY ** 2)) ** .5
                if vectLen != 0:
                    moveX = float(moveX) / vectLen
                    moveY = float(moveY) / vectLen
                self.playerPos[0] += moveX * self.playerSpeed * self.dt
                self.playerPos[1] += moveY * self.playerSpeed * self.dt
                
                
                
                #
                #                       ball physics (rigid tether)
                #
                tetherVect = [self.playerPos[0] - self.ballPos[0], self.playerPos[1] - self.ballPos[1]]
                allowedNotNormalized = [tetherVect[1], -tetherVect[0]] 
                allowedVect = [allowedNotNormalized[0] / self.tetherLen, allowedNotNormalized[1] / self.tetherLen]
                #calculate direction ball *wants* to move in
                self.ballVel = [self.ballPos[0] - self.prevPos[0], self.ballPos[1] - self.prevPos[1]]
                self.ballVel[1] += self.grav
                #project the ball's velocity onto the allowed direction
                self.mult *= self.entropy
                self.mult += (self.ballVel[0] * allowedVect[0] + self.ballVel[1] * allowedVect[1])/(allowedVect[0] ** 2 + allowedVect[1] ** 2)
                ballMoveVect = [allowedVect[0]*self.mult, allowedVect[1]*self.mult]
                #update ball position
                self.prevPos = self.ballPos
                self.ballPos[0] += ballMoveVect[0] * self.dt
                self.ballPos[1] += ballMoveVect[1] * self.dt
                #keep tether same length
                dist = math.sqrt((self.playerPos[0] - self.ballPos[0])**2 + (self.playerPos[1] - self.ballPos[1])**2)
                if (dist != self.tetherLen):
                    self.ballPos[0] += (tetherVect[0]/self.tetherLen)*(dist-self.tetherLen)
                    self.ballPos[1] += (tetherVect[1]/self.tetherLen)*(dist-self.tetherLen)
                

                if self.printScreen == True:
                    
                    #                           draw the current frame
                    # 
                    self.screen.fill("black") #overrite previous frame
                    pygame.draw.circle(self.screen, self.playerColor, (self.playerPos[0], self.playerPos[1]), self.playerRad)
                    pygame.draw.circle(self.screen, self.playerColor, (self.ballPos[0], self.ballPos[1]), self.ballRad)
                    pygame.draw.line(self.screen, self.playerColor, (self.ballPos[0], self.ballPos[1]), (self.playerPos[0], self.playerPos[1]), self.tetherWid)
                    pygame.display.flip() # TODO: figure out what this function


                #disable speed scaling for framerates
                # self.dt = self.clock.tick(self.maxfps) / 1000 #update dt as the num of seconds since last frame
                # self.dt *= self.gameSpeed

            # print some stats, for overseeing training
            accuracy = float(self.agent.memory_buffer_reward.__len__()) / float((self.agent.memory_buffer_reward.__len__() + self.agent.memory_buffer.__len__()))
            accuracy *= 100
            if (self.averageMemory.__len__() < self.memBuffer):
                self.averageMemory.append(accuracy)
            else:
                self.averageMemory[self.numEpisodes % self.memBuffer] = accuracy
            episodeAvg = float(sum(self.averageMemory)) / float(self.averageMemory.__len__())
            with open(self.trialName+"/LOG.txt", "a+") as f:
                f.write("accuracy: "+str(round(accuracy, 3))+"  in episode "+ str(self.numEpisodes)+" and  "+str(round(episodeAvg, 3))+" in last "+ str(self.averageMemory.__len__())+" episodes   random prob: "+ str(round(self.agent.getProb() * 100, 3))+"\n")
            if (self.agent.getProb() < 0.01):
                self.agent.disableRandom()


            #Saving the model's weights
            if (self.numEpisodes % self.saveFrequency == 0):
                self.numsave += 1
                checkpoint_path = "./"+self.trialName+"/checkpoint"+str(self.numsave)
                self.agent.model.save_weights(checkpoint_path)
                with open(self.trialName+"/LOG.txt", "a+") as f:
                    f.write("Saved! -- checkpoint "+ str(self.numsave)+"\n")

            self.agent.train()
            self.agent.update_exploration_probability()
        
game = Game()
game.simulation()