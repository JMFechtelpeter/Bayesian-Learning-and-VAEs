# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 10:29:18 2020

@author: Janik
"""

import numpy as np
import matplotlib.pyplot as plt
import hidden_markov_lib2 as hm
#from matplotlib import rc

plt.style.use('seaborn')
plt.rcParams['text.usetex'] = False
plt.rcParams['svg.fonttype'] = 'none'
"""
Plot 1: Sanity Check: lernen die Algorithmen einen Pfad, den sie ganz oft zu Gesicht bekommen? p_observation = p_transition = 0.75
"""

plt.rcParams['figure.figsize'] = (15,5)

while True:
    G = hm.gridworld(4,4,p_trans=0.75,p_observe=0.75)
    
    G.explore(10)
    G.O = G.O*100
    
    F_bayesian_Qs_it1 = G.free_energy(G.learning_bp(iterations=1))
    F_bayesian_Qs_it4 = G.free_energy(G.learning_bp(iterations=4))
    
    plt1 = plt.subplot(1,3,1)
    plt.plot(F_bayesian_Qs_it1,color='orange')
    plt.plot(F_bayesian_Qs_it4,color='red')
    plt.xlabel('HMM sessions')
    plt.ylabel('KL divergence')
    plt.legend(('bayesian 1 iterations','bayesian 4 iterations'))
    
    limits = plt.gca().get_ylim()
    
    F_VAE_full_Qs_batch = G.free_energy(G.VAE_bp(stepwise=False,VAE_type='full'))
    F_VAE_full_Qs_stepwise = G.free_energy(G.VAE_bp(stepwise=True,VAE_type='full'))
    
    plt2 = plt.subplot(1,3,2)
    plt.plot(F_VAE_full_Qs_batch,color='green')
    plt.plot(F_VAE_full_Qs_stepwise,color='blue')
    plt.xlabel('HMM sessions')
    plt.ylabel('KL divergence')
    plt.legend(('full VAE','full stepwise VAE','MC VAE','MC stepwise VAE'))
    
#    limits = plt.gca().get_ylim()
    
    limits = [min([limits[0],plt.gca().get_ylim()[0]]), max([limits[1],plt.gca().get_ylim()[1]])]
    
    F_VAE_MC_Qs_batch = G.free_energy(G.VAE_bp(stepwise=False,VAE_type='MC'))
    F_VAE_MC_Qs_stepwise = G.free_energy(G.VAE_bp(stepwise=True,VAE_type='MC'))
    
    plt3 = plt.subplot(1,3,3)
    plt.plot(F_VAE_MC_Qs_batch,color='magenta')
    plt.plot(F_VAE_MC_Qs_stepwise,color='purple')
    plt.xlabel('HMM sessions')
    plt.ylabel('Free Energy minus surprise')
    plt.legend(('MC VAE','MC stepwise VAE'))
    
    limits = [min([limits[0],plt.gca().get_ylim()[0]]), max([limits[1],plt.gca().get_ylim()[1]])]
    
    plt1.set_ylim(limits)
    plt2.set_ylim(limits)
    plt3.set_ylim(limits)
    
    plt.savefig('D:\Studium\MA Bachelorarbeit\Grafiken\plot1.svg')
    plt.show()
    
    answer = input('OK? (y/n)')
    
    if answer == 'y':
        break


"""
Plot 2: F im Verlauf einer echten Lern-Session, p_observation = p_transition = 0.75
"""

plt.rcParams['figure.figsize'] = (15,5)
prior = np.zeros(16)
prior[0] = 1

while True:

    G = hm.gridworld(4,4,p_trans=1,p_observe=0.75,prior=prior)
    
    for session in range(100):
        G.explore(10,reset=True)
    
    F_bayesian_Qs_it1 = G.free_energy(G.learning_bp(iterations=1))
    F_bayesian_Qs_it4 = G.free_energy(G.learning_bp(iterations=4))
    
    plt1 = plt.subplot(1,3,1)
    plt.plot(F_bayesian_Qs_it1,color='orange')
    plt.plot(F_bayesian_Qs_it4,color='red')
    plt.xlabel('HMM sessions')
    plt.ylabel('Free Energy minus surprise')
    plt.legend(('bayesian 1 it.','bayesian 4 it.'))
    
    limits = plt.gca().get_ylim()
    
    F_VAE_full_Qs_batch = G.free_energy(G.VAE_bp(stepwise=False,VAE_type='full'))
    F_VAE_full_Qs_stepwise = G.free_energy(G.VAE_bp(stepwise=True,VAE_type='full'))
    
    plt2 = plt.subplot(1,3,2)
    plt.plot(F_VAE_full_Qs_batch,color='green')
    plt.plot(F_VAE_full_Qs_stepwise,color='blue')
    plt.xlabel('HMM sessions')
    plt.ylabel('Free Energy minus surprise')
    plt.legend(('full VAE','full stepwise VAE'))
    
    limits = [min([limits[0],plt.gca().get_ylim()[0]]), max([limits[1],plt.gca().get_ylim()[1]])]
    
    F_VAE_MC_Qs_batch = G.free_energy(G.VAE_bp(stepwise=False,VAE_type='MC'))
    F_VAE_MC_Qs_stepwise = G.free_energy(G.VAE_bp(stepwise=True,VAE_type='MC'))
    
    plt3 = plt.subplot(1,3,3)
    plt.plot(F_VAE_MC_Qs_batch,color='magenta')
    plt.plot(F_VAE_MC_Qs_stepwise,color='purple')
    plt.xlabel('HMM sessions')
    plt.ylabel('Free Energy minus surprise')
    plt.legend(('MC VAE','MC stepwise VAE'))
    
    limits = [min([limits[0],plt.gca().get_ylim()[0]]), max([limits[1],plt.gca().get_ylim()[1]])]

    plt1.set_ylim(limits)
    plt2.set_ylim(limits)
    plt3.set_ylim(limits)
    
    plt.savefig('D:\Studium\MA Bachelorarbeit\Grafiken\plot2.svg')
    plt.show()
    
    answer = input('OK? (y/n)')
    
    if answer == 'y':
        break

"""
Plot 3: Durchschnittliches F in Abhängigkeit von p_observation = 0.5, 0.75, 1, p_transition = 0.5, 0.75, 1
"""

plt.rcParams['figure.figsize'] = (15,10)

while True:
    p_trans = np.array([0.5,0.75,1])
    p_observe = np.array([0.5,0.75,1])
    plots = []
        
    index = 1
    for pt in p_trans:
        for po in p_observe:
    
            G = hm.gridworld(4,4,p_trans=pt,p_observe=po)
            
            for session in range(100):
                G.explore(10,reset=True)
                
            F = np.zeros((6,100))
            F[0] = G.free_energy(G.learning_bp(iterations=1))
            F[1] = G.free_energy(G.learning_bp(iterations=4))
            F[2] = G.free_energy(G.VAE_bp(stepwise=False,VAE_type='full'))
            F[3] = G.free_energy(G.VAE_bp(stepwise=True,VAE_type='full'))
            F[4] = G.free_energy(G.VAE_bp(stepwise=False,VAE_type='MC'))
            F[5] = G.free_energy(G.VAE_bp(stepwise=True,VAE_type='MC'))
            
#            mean_F = np.zeros(6)
#                
#            mean_F[0] = np.mean(G.free_energy(G.learning_bp(iterations=1)))
#            mean_F[1] = np.mean(G.free_energy(G.learning_bp(iterations=4)))
#            
#            mean_F[2] = np.mean(G.free_energy(G.VAE_bp(stepwise=False,VAE_type='full')))
#            mean_F[3] = np.mean(G.free_energy(G.VAE_bp(stepwise=True,VAE_type='full')))
#            mean_F[4] = np.mean(G.free_energy(G.VAE_bp(stepwise=False,VAE_type='MC')))
#            mean_F[5] = np.mean(G.free_energy(G.VAE_bp(stepwise=True,VAE_type='MC')))
            
            plots.append(plt.subplot(3,3,index))
            plt.boxplot(F.transpose())
            plt.xticks(ticks=(1,2,3,4,5,6),labels=('BL1','BL4','VF','VFS','VM','VMS'))
#            plt.bar(('BL1','BL4','VF','VFS','VM','VMS'),mean_F,color=['orange','red','green','blue','magenta','purple'])
#            plt.title('p_trans='+str(pt)+' p_obs='+str(po))
            
            if index==1:
                limits = plt.gca().get_ylim()
            else:
                limits = [min([limits[0],plt.gca().get_ylim()[0]]), max([limits[1],plt.gca().get_ylim()[1]])]
            
            index+=1
            
    for plot in plots:
        plot.set_ylim(limits)
    
    plt.savefig('D:\Studium\MA Bachelorarbeit\Grafiken\plot3.svg')
    plt.show()
    
    answer = input('OK? (y/n)')
    
    if answer == 'y':
        break


"""
Plot 4: Rechenzeit in Abhängigkeit von p_observation = 0.5, 0.75, 1, p_transition = 0.5, 0.75, 1
"""

plt.rcParams['figure.figsize'] = (15,10)

while True:
    p_trans = np.array([0.5,0.75,1])
    p_observe = np.array([0.5,0.75,1])
    plots = []
    
    index = 1
    for pt in p_trans:
        for po in p_observe:
    
            G = hm.gridworld(4,4,p_trans=pt,p_observe=po)
            
            for session in range(100):
                G.explore(10,reset=True)
                
            duration = np.zeros(6)
                
            duration[0], _ = G.measure_perf(G.learning_bp,iterations=1)
            duration[1], _ = G.measure_perf(G.learning_bp,iterations=4)
            
            duration[2], _ = G.measure_perf(G.VAE_bp,stepwise=False,VAE_type='full')
            duration[3], _ = G.measure_perf(G.VAE_bp,stepwise=True,VAE_type='full')
            duration[4], _ = G.measure_perf(G.VAE_bp,stepwise=False,VAE_type='MC')
            duration[5], _ = G.measure_perf(G.VAE_bp,stepwise=True,VAE_type='MC')
            
            plots.append(plt.subplot(3,3,index))
            plt.bar(('BL1','BL4','VF','VFS','VM','VMS'),duration,color=['orange','red','green','blue','magenta','purple'])
            plt.title('p_trans='+str(pt)+' p_obs='+str(po))
            
            if index==1:
                limits = plt.gca().get_ylim()
            else:
                limits = [min([limits[0],plt.gca().get_ylim()[0]]), max([limits[1],plt.gca().get_ylim()[1]])]
            
            index+=1
    
    for plot in plots:
        plot.set_ylim(limits)
    
    plt.savefig('D:\Studium\MA Bachelorarbeit\Grafiken\plot4.svg')
    plt.show()
    
    answer = input('OK? (y/n)')
    
    if answer == 'y':
        break
    

"""
Plot 5: Beispiel-q für einen BL und einen VAE am Ende von 100 Sessions
"""

plt.rcParams['figure.figsize'] = (15,10)
cmap = 'rainbow'

while True:
    
    G = hm.gridworld(4,4,p_trans=0.75,p_observe=1,prior=prior)
    
    for session in range(100):
        G.explore(6,reset=True)
        
    true_p = G.bp_belief(O=G.O[-1])
    bayesian_q = G.learning_bp(O=[G.O[-1]],iterations=4)
    VAE_q = G.VAE_bp(O=G.O[-1],VAE_type='full',stepwise=True)

    axes = []
    fig = plt.figure()
    
    axes.append(fig.add_subplot(1,3,1))
    plot = G.plot_pdf(true_p[5],show=False,cmap=cmap)
    axes[0].set_title('True posterior')
    
    axes.append(fig.add_subplot(1,3,2))
    plot = G.plot_pdf(bayesian_q[-1][5],show=False,cmap=cmap)
    axes[1].set_title('BL4')
    
    axes.append(fig.add_subplot(1,3,3))
    plot = G.plot_pdf(VAE_q[-1][5],show=False,cmap=cmap)
    axes[2].set_title('VF')
    
    fig.colorbar(plot,ax=axes,fraction=0.015)
    
    plt.savefig('D:\Studium\MA Bachelorarbeit\Grafiken\plot5.svg')
    plt.show()
    
    answer = input('OK? (y/n/cmap)')
    
    if answer == 'y':
        break
    elif answer != 'n':
        cmap = answer
    