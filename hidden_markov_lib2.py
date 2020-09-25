# -*- coding: utf-8 -*-
"""
Created on Wed May  6 19:43:54 2020

@author: Janik
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma
from discrete_lib import draw, normalize, dkl
import vae_lib as vae
from time import process_time

class hidden_markov:
#Base class for a Hidden Markov Process,
    
    def __init__(self,size,transition,observation):
    #Initialize hidden markov process with fixed state vector length, and constant transition and observation probability

        self.size = np.int32(size)

        self.transition = transition
        self.observation = observation
        
    def one_hot(self,position):
    #Returns a vector of zeros with 1 at <position>
        x = np.zeros(self.size)
        x[position] = 1
        
        return x
    
    def draw_state(self,p):
    #Draws a state vector from a categorical distribution <p>
        s0 = draw(p)
        return self.one_hot(s0)
        
    def move(self,s):
    #Given a state <s>, draws the next state from the transition probability
        pdf = np.dot(s,self.transition)
        newpos = draw(pdf)
        return self.one_hot(newpos)
        
    def observe(self,s):
    #Given a state <s>, draws an observation from the observation probability
        pdf = np.dot(s,self.observation)
        obspos = draw(pdf)       
        return self.one_hot(obspos)
    
    def process(self,s0,length):
    #Given a state, draws <length> subsequent states and observations
        S = np.zeros([length,self.size]) 
        O = np.zeros([length,self.size]) 
        
        S[0] = s0
        O[0] = self.observe(s0)
        
        for t in range(1,length):
            S[t] = self.move(S[t-1])
            O[t] = self.observe(S[t])
            
        return S, O
    
    def p_observation(self,O):
    #returns a matrix of observation probabilities, where rows correspond to given observations <O> and columns to states.
        p = np.dot(self.observation,O.transpose())
        return p.transpose()
    
    
class hidden_markov_agent(hidden_markov):
    
    def __init__(self,size,transition,observation,prior):
    #Base class for inference with the Hidden Markov Model        
        super().__init__(size,transition,observation)
        self.p0 = prior
        self.state = self.draw_state(self.p0)
        self.S = []
        self.O = []
        
    def measure_perf(self,function,repetitions=1,*args,**kwargs):
    #returns calculation time for <function> with arguments <*args,**kwargs>, calles <repetition> times        
        tick = process_time()
        for i in range(repetitions):
            result = function(*args,**kwargs)
        tock = process_time() - tick
        
        return tock, result       
    
    def free_energy(self,Q):
    #calculated DKL between <Q> and the true posterior for every session, if Q is a list of all observations, and for the last session, if Q is a list of T observations.
        if not isinstance(Q,list):
            Q = [Q]
            O = [self.O[-1]]
        elif len(Q) != len(self.O):
            raise BaseException('Q and O are not of the same length')
        else:
            O = self.O
            
        DKL = np.zeros(len(O))
        
        for session in range(len(O)):
            true_posterior = self.bp_belief(O=O[session])
            for t in range(O[session].shape[0]):
                DKL[session] += dkl(Q[session][t],true_posterior[t])
                
        return DKL           
        
        
    def explore(self,length,reset=True):
    #Starting from self.state, draws <length> subsequent states and observations and saves them. Sets self.state to the last state.
        if reset:
            self.state = self.draw_state(self.p0)
        
        S, O = self.process(self.state,length)
        
        if reset:
            self.S.append(S)
            self.O.append(O)
        else:
            self.S[-1] = np.vstack(self.S[-1],S)
            self.O[-1] = np.vstack(self.O[-1],O)
            
        self.state = S[-1]
        
    def forget(self):
    #Resets the agent's state and observation memory.
        self.S = []
        self.O = []            

    ''' Methods to calculate the exact posterior without variational inference    
    '''
    
    def posterior(self,O=np.array([])):
    #returns true posteriors over states s_1,...,s_t given observations <o_1,...,o_t>
        
        if O.shape == (0,):
            O = self.O[-1]
        
        aux_prior = np.zeros(O.shape)
        aux_prior[0] = self.p0
        for k in np.arange(1,O.shape[0]):
            aux_prior[k] = np.dot(self.p_observation(O[k-1]) * aux_prior[k-1], self.transition)
            
        p_sk = aux_prior * self.p_observation(O)
        
        p_ot = np.ones(O.shape)
        for l in np.flip(np.arange(O.shape[0]-1)):
            p_ot[l] = np.dot(self.transition, self.p_observation(O[l+1]) * p_ot[l+1])
            
        return normalize(p_sk * p_ot, axis=1)
            
        
    
    """
        Mean-field belief calculation
    """
    
    def mf_belief(self,O=np.array([]),future=0,iterations=1,init='prior',return_Qo=False):
    #returns mean field belief over states and observations 1,...,t+<future>, given observations <o_1,...,o_t>
    #if no observations were given, O is set to the last session
    #<init> sets initialization condition on the belief over states.
    #<iterations> lets the algorithm iterate over all states and observations.
        
        if O.shape==(0,):
            O = self.O[-1]
            
        T = O.shape[0] + future
        
        if init=='zero':
            Qs = np.zeros([T,O.shape[1]])
            t0 = 0
        elif init=='prior':
            Qs = np.zeros([T,O.shape[1]])
            Qs[0] = self.p0
            t0 = 1
        elif init=='uniform':
            Qs = np.ones([T,O.shape[1]]) / self.size
            t0 = 0
        else:
            raise BaseException("In mf_belief: input 'init' must have one of the following values: 'zero', 'prior', or 'uniform'")
        
        Qo = np.zeros([T,O.shape[1]],dtype=np.float64)        
        obslen = np.min([T,O.shape[0]])
        Qo[0:obslen] = O[0:obslen]        

        lnP = np.nan_to_num(np.log(self.transition))
        lnO = np.nan_to_num(np.log(self.observation))
        
        for i in range(iterations):
            if i>0:
                t0 = 0
                
            for t in range(t0,T):

                buffer = np.dot(lnO,Qo[t])
                if t>0:
                    buffer += np.dot(Qs[t-1], lnP)
                if t<T-1:
                    buffer += np.dot(lnP, Qs[t+1])
                Qs[t] = normalize(np.exp(buffer))                                                            
        
                if t>=obslen:
                    Qo[t] = normalize(np.exp(np.dot(lnO, Qs[t])))                        
   
        if return_Qo:
            return Qs, Qo
        else:
            return Qs
    
    
    """
        Belief Propagation
    """

    def bp_belief(self,O=np.array([]),future=0,return_Qo=False):
    #returns Bethe beliefs calculated via Belief propagation (they are equivalent) over states and observations 1,...,t+<future>, given observations <o_1,...,o_t>
    #if no observations were given, O is set to the last session
        
        if O.shape==(0,):
            O = self.O[-1]            
        T = O.shape[0] + future
        
        Qs = np.zeros([T,O.shape[1]])        
        Qo = np.zeros([T,O.shape[1]],dtype=np.float64)        
        obslen = np.min([T,O.shape[0]])
        Qo[0:obslen] = O[0:obslen]
            
        Mbackward = np.ones([T,self.size])
        Mforward = np.ones([T,self.size])
        Mtoobs = np.ones([T,self.size])
        Mfromobs = np.ones([T,self.size])
        
        Mfromobs[0:obslen] = self.p_observation(O[0:obslen])
        
        for j in range(T-2,-1,-1):                
            Mbackward[j] = np.dot(self.transition,Mfromobs[j+1]*Mbackward[j+1])
            
        Mforward[0] = self.p0       
        Qs[0] = normalize(Mforward[0] * Mbackward[0] * Mfromobs[0])
        
        for j in range(1,T):
            Mforward[j] = np.dot(Mfromobs[j-1]*Mforward[j-1],self.transition)
            Qs[j] = normalize(Mforward[j] * Mbackward[j] * Mfromobs[j])
               
        for j in range(obslen,T):
            Mtoobs[j] = np.dot(Mforward[j]*Mbackward[j],self.observation)
            Qo[j] = normalize(Mtoobs[j])
            
        if return_Qo:
            return Qs, Qo
        else:
            return Qs
    
    """
        Belief Propagation with Bayesian Learning
    """
    
    def learning_bp(self,O=np.array([]),iterations=1,return_alpha=False):
    #Returns beliefs over states given observations <O> calculated with the Bayesian Learning algorithm, without utilizing the observation probability.
    #if no O is given, it is set to the complete Observation memory.
    #<return_alpha> decides whether to return the concentration parameters of state-observation combinations.
        
        if not isinstance(O,list):
            if O.shape==(0,):
                O = self.O
            else:
                O = [O]
            
        Qs = []        
        alpha = np.ones((self.size,self.size))
#        acc_F = np.zeros(len(O))
        
        for session in range(len(O)):
            
            O_buffer = O[session]
            
            Mforward = np.ones(O_buffer.shape)
            Q_buffer = np.ones(O_buffer.shape)
            mask = O_buffer[0]==1
            Mforward[0] = self.p0
            
            for it in range(iterations):
                Q_buffer[0] = normalize(Mforward[0] * np.exp(digamma(alpha[:,mask].squeeze()) - digamma(alpha.sum(axis=1))))
                alpha[:,mask] += np.expand_dims(Q_buffer[0],1)
            
            for t in range(1,O_buffer.shape[0]):
                
                Mfromobs = alpha[:,mask].squeeze() / np.sum(alpha[:,mask])
                Mforward[t] = np.dot(Mfromobs*Mforward[t-1],self.transition)
                mask = O_buffer[t]==1
                
                for it in range(iterations):
                    Q_buffer[t] = normalize(Mforward[t] * np.exp(digamma(alpha[:,mask].squeeze()) - digamma(alpha.sum(axis=1))))
                    alpha[:,mask] += np.expand_dims(Q_buffer[t],1)
                    
            Qs.append(Q_buffer)
            
        if return_alpha:
            return Qs, alpha
        else:
            return Qs
    
    """
        Belief Propagation with VAE
    """
    
    def VAE_bp(self,O=np.array([]),DKL_weight=1,VAE_type='full',stepwise=False,return_loss=False):
    #Returns beliefs over states given observations <O> calculated with a Variational Autoencoder, without utilizing the observation probability
    #if no O is given, it is set to the complete Observation memory.
    #<VAE_type> can be set to 'full' to activate full integration, or to 'MC' to activate Monte Carlo integration
    #<stepwise> set to True lets gradient descent be run after every observation, <stepwise> set to False lets gradient descent be run after every session.
    #<return_loss> decides whether to return the VAE's loss over sessions.
        
        if not isinstance(O,list):
            if O.shape==(0,):
                O = self.O
            else:
                O = [O]
            
        Qs = []
        
        if VAE_type=='full':
            model = vae.VAE_full(self.size,self.size,self.size)
            loss_function = vae.full_loss
        elif VAE_type=='MC':
            model = vae.VAE_MC(self.size,self.size,self.size)
            loss_function = vae.MC_loss
        else:
            raise BaseException("VAE_type must be specified as 'full' or 'MC'.")
            
        optimizer = vae.optim.SGD(model.parameters(),lr=0.1)    #This is stochastic gradient descent
        acc_loss = vae.torch.zeros(len(O),requires_grad=False)
        
        for session in range(len(O)):
            
            O_buffer = O[session]
            Mforward = np.ones(O_buffer.shape)
            obs = vae.torch.from_numpy(O_buffer).float().view(-1,self.size)
            
            if stepwise:
                
                Q_buffer = np.zeros(O_buffer.shape)
                
                Mforward[0] = self.p0                
                for t in range(O_buffer.shape[0]):
                    
                    optimizer.zero_grad()
                    
                    current_obs = obs[t].view(1,-1)
                    unnormalized_log_p, unnormalized_log_q = model(current_obs)
                    msg = vae.torch.from_numpy(Mforward[t]).float()
                    
                    loss = loss_function(current_obs,unnormalized_log_p,unnormalized_log_q,msg,size=self.size,DKL_weight=DKL_weight)

                    loss.backward()
                    optimizer.step()

                    with vae.torch.no_grad():
                        
                        acc_loss[session] += loss
                        Q_buffer[t] = vae.F.softmax(unnormalized_log_q,dim=1).numpy()
                        
                        if VAE_type=='full':
                            p = vae.F.softmax(unnormalized_log_p,dim=1)
                            Mfromobs = vae.F.softmax(p[:,O_buffer[t]==1],dim=0).numpy().squeeze()
                        else:
                            Mfromobs = np.ones(self.size)
                        
                    if t<O_buffer.shape[0]-1:
                        Mforward[t+1] = normalize(np.dot(Mfromobs*Mforward[t],self.transition))
                        
                Qs.append(Q_buffer)

            else:            
                
                optimizer.zero_grad()
            
                unnormalized_log_p, unnormalized_log_q = model(obs)
                            
                Mforward[0] = self.p0
                for t in range(1,O_buffer.shape[0]):
                    if VAE_type=='full': 
                        with vae.torch.no_grad():
                            p = vae.F.softmax(unnormalized_log_p,dim=1)
                            Mfromobs = vae.F.softmax(p[:,O_buffer[t]==1],dim=0).numpy().squeeze()
                    else:
                        Mfromobs = np.ones(self.size)
                    Mforward[t] = normalize(np.dot(Mfromobs*Mforward[t-1],self.transition))            
                msg = vae.torch.from_numpy(Mforward).float()            
            
                loss = loss_function(obs,unnormalized_log_p,unnormalized_log_q,msg,size=self.size,DKL_weight=DKL_weight)
            
                with vae.torch.no_grad():
                    acc_loss[session] = loss
                    Qs.append(vae.F.softmax(unnormalized_log_q,dim=1).numpy())
                    
                loss.backward()
                optimizer.step()
            
        if return_loss:
            return Qs, acc_loss
        else:
            return Qs            
        


class gridworld(hidden_markov_agent):
#Supclass for the Gridworld example of the HMM.
    
    def __init__(self,height,width,p_trans=0.5,p_observe=1,
                 transition=np.array([]),observation=np.array([]),prior=np.array([]),roundtrip=False):
    #Initialize gridworld with fixed width and height, and constant transition and observation probabilities
    #<roundtrip> allows for the agent to go backwards as well, useful to prolongue the session
    
        self.width = np.int32(width)
        self.height = np.int32(height)
        size = np.int32(width*height)        
                
        if len(transition.shape) != 2:
            print('Using predefined transition matrix' + roundtrip*' roundtrip')
            transition = np.ones([size,size]) * 1e-5
            
            #Transition Matrix according to section 4.3
            for i in range(size):
                if roundtrip:
                    N = self.neighborhood(i)
                else:
                    N = self.neighborhood(i)[0:3:2]
                N = N[N>=0]
                
                transition[i,N] = p_trans / (2 + roundtrip*2)               #movement               
                transition[i,i] = 1 - (p_trans/(2 + roundtrip*2))*(len(N))  #staying in the same field
                
                transition[i] = normalize(transition[i])
        
        
        if len(observation.shape) != 2:
            print('Using predefined observation matrix')
            observation = np.zeros([size,size])

            #Observation Matrix according to section 4.3
            for i in range(size):
                observation[i,i] = p_observe
                
                N = self.neighborhood(i)
                observation[i,N[N>=0]] = (1 - p_observe)/np.sum(N>=0)
                
                observation[i] += 1e-5
                
                observation[i] = normalize(observation[i])
                
        if len(prior.shape) != 1 or prior.size != size:
            #Prior according to section 4.3
            print('Using predefined prior')
            prior = np.zeros(size)
            prior[0] = prior[1] = prior[self.width] = prior[self.width+1] = 0.25
            
        super().__init__(size,transition,observation,prior)
                                
                    
    def neighborhood(self,pos):
    #returns the neighboring positions of <pos>
        N = np.ones(4,dtype=np.int32) * (-1)
        c = self.coord(pos)
        
        if c[0] < self.height-1:
            N[0] = np.int32(pos+self.width)    #top neighbor
        if c[0] > 0:
            N[1] = np.int32(pos-self.width)    #bottom neighbor
        if c[1] < self.width-1:
            N[2] = np.int32(pos+1)             #right neighbor
        if c[1] > 0:
            N[3] = np.int32(pos-1)             #left neighbor
            
        return N       
        
    def coord(self,pos):        
    #transforms <pos> into 2D Gridworld coordinates
        col = np.int32(pos % self.width)
        row = np.int32(np.floor(pos/self.width))
        
        return row,col
    
    def pos(self,*coord):
    #transforms <coord> into 1D position
        return np.int32(coord[0]*self.width + coord[1])    
    
    def coordstate(self,state=[]):
    #returns <state> or the current state as 2D coordinates
        if np.sum(np.shape(state))==0:
            state = self.state
        pos = np.int32(np.dot(state,np.arange(self.size)))
        return self.coord(pos)
    
    def gridstate(self,state=[]):
    #returns <state> or the current state as a Gridworld matrix
        return state.reshape([self.height,self.width])
        

    def plotstate(self,state=[]):
    #plots <state> or the current state
        if np.sum(np.shape(state))==0:
            state = self.state
            
        X,Y = np.meshgrid(np.arange(self.width),np.arange(self.height))
        plt.plot(X,Y,marker='o',color='black',linestyle='none')
        plt.plot(self.coordstate(state)[1],self.coordstate(state)[0],marker='o',color='red')
        plt.show()
        
    def plot_pdf(self,pdf,show=True,cmap='rainbow',log=False):
    #plots a probability distribution <pdf> over the Gridworld.
    #Set <show> to false to avoid showing the figure in the console, useful because plt.show() wipes the figure clean and impedes saving
    #Specify the probability colormap by <cmap>, set <log> to true to show log probabilities
        if log:
            fig = plt.imshow(np.log(self.gridstate(pdf)+1e-10),origin='lower',cmap=cmap,vmax=0)
        else:
            fig = plt.imshow(self.gridstate(pdf),origin='lower',cmap=cmap,vmin=0,vmax=1)
            
        if show:
            plt.show()   
            
        return fig
    
    def plotprocess(self,S=np.array([]),O=np.array([])):
    #plots a sequence of states <S> and observations <O>, if left empty, S and O are set to the last session.
        if np.sum(np.shape(S))==0:
            S = self.S[-1]
        if np.sum(np.shape(O))==0:
            O = self.O[-1]
            
        X,Y = np.meshgrid(np.arange(self.width),np.arange(self.height))
        plt.plot(X,Y,marker='o',color='black',linestyle='none')
        
        coordproc = self.coord(np.dot(S,np.arange(self.size)))
        coordobs = self.coord(np.dot(O,np.arange(self.size)))

        plt.plot(coordproc[1],coordproc[0],linewidth=5)
        plt.plot(coordobs[1],coordobs[0],linestyle='dashed',linewidth=3)
        plt.show()
        
