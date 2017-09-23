__author__ = 'sudab'

import numpy as np
import copy
import matplotlib.pyplot as plt

def safe_ln(x,out):
    if x <= 0:
        return out
    else:
        return np.log(x)

def safe_div(x,y,out):
    if y!=0:
        return x/y
    else:
        return out

def safe_exp(x,out):
    if x>200:
        return out
    else:
        return np.exp(x)

def calc_mu(K,t,mu,mu_states,q,qinit,gwg,mdp):
    for s_new in set(gwg.states) - set(gwg.edges):
        (y,x) = gwg.coords(s_new)
        for u in range(gwg.nactions):
            if K == 0:
                mu[K,t+1,x,y,u] = sum([mdp.prob_delta(s_old,u_old,s_new)*
                                       qinit[gwg.coords(s_old)[1],gwg.coords(s_old)[0],u]*
                                       mu[K,t,gwg.coords(s_old)[1],gwg.coords(s_old)[0],u_old]
                                       for (s_old,u_old) in mdp.pre(s_new)])
            else:
                mu[K,t+1,x,y,u] = sum([mdp.prob_delta(s_old,u_old,s_new)*
                                       q[K-1,t,gwg.coords(s_old)[1],gwg.coords(s_old)[0],u_old,u]*
                                       mu[K,t,gwg.coords(s_old)[1],gwg.coords(s_old)[0],u_old]
                                       for (s_old,u_old) in mdp.pre(s_new)])
        mu_states[y,x] = sum(mu[K,t,x,y,:])
    return mu,mu_states

def calc_nu(K,t,mu,nu,q,qinit,gwg):
    for u_old in range(gwg.nactions):
        for u in range(gwg.nactions):
            if K == 0:
                nu[K,t,u_old,u] = sum([qinit[gwg.coords(s_old)[1],gwg.coords(s_old)[0],u]*
                                    safe_div(mu[K,t,gwg.coords(s_old)[1],gwg.coords(s_old)[0],u_old],
                                            sum([mu[K,t,gwg.coords(s_old2)[1],gwg.coords(s_old2)[0],u_old]
                                            for s_old2 in set(gwg.states) - set(gwg.edges)]),0.1)
                                    for s_old in set(gwg.states) - set(gwg.edges)])
            else:
                nu[K,t,u_old,u] = sum([q[K-1,t,gwg.coords(s_old)[1],gwg.coords(s_old)[0],u_old,u]*
                                        safe_div(mu[K,t,gwg.coords(s_old)[1],gwg.coords(s_old)[0],u_old],
                                            sum([mu[K,t,gwg.coords(s_old2)[1],gwg.coords(s_old2)[0],u_old]
                                            for s_old2 in set(gwg.states) - set(gwg.edges)]),0.1)
                                        for s_old in set(gwg.states) - set(gwg.edges)])
        nu[K,t,u_old,:] = nu[K,t,u_old,:]/sum(nu[K,t,u_old,:])
    return nu


def alg_m0n1(gwg,mdp,iter,T,beta,cost,plot,pltiter):
    # Initialize parameters
    psi_max = beta*10000
    rho_max = beta*10000
    initcoords = tuple(reversed(gwg.coords(gwg.current[0])))
    targcoords = tuple(reversed(gwg.coords(gwg.targets[0][0])))
    # Define variables
    mu = np.full((iter,T+1,gwg.ncols,gwg.nrows,gwg.nactions),0.0)
    nu = np.full((iter,T,gwg.nactions,gwg.nactions),0.0)
    rho = np.full((iter,T,gwg.ncols,gwg.nrows,gwg.nactions,gwg.nactions),0.0)
    psi = np.full((iter,T+1, gwg.ncols,gwg.nrows,gwg.nactions),0.0)  #-log(phi)
    q = np.full((iter,T,gwg.ncols,gwg.nrows,gwg.nactions,gwg.nactions),0.0)

    # Initialize variables
    qinit = np.full((gwg.ncols,gwg.nrows,gwg.nactions),1.0/gwg.nactions)
    for K in range(iter):
        for u in range(gwg.nactions):
            mu[K,0,initcoords[0],initcoords[1],u] = 1.0/(gwg.nactions) #initial distribution
        for s in gwg.states:
            if tuple(reversed(gwg.coords(s))) != targcoords:
                psi[K,T,gwg.coords(s)[1],gwg.coords(s)[0],:] = np.full(gwg.nactions,100) # Terminal cost

    mdp._prepare_post_cache()
    mdp._prepare_pre_cache()


    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # im = ax.imshow(np.full((gwg.nrows,gwg.ncols),0),cmap = 'hot',interpolation='nearest')
    # plt.show(block=False)
    plt.ion()
    for K in range(iter):
        print 'At iteration ', K+1, '/', iter
        # Forward path
        print 'Forward path:'
        mu_states = np.full((gwg.nrows,gwg.ncols),0.0)
        for t in range(T):
            print 'At timestep ', t+1, '/', T
            mu,mu_states = calc_mu(K,t,mu,mu_states,q,qinit,gwg,mdp) # Calculate mu
            nu = calc_nu(K,t,mu,nu,q,qinit,gwg) # Calculate nu


            if plot and np.mod(K/pltiter,pltiter) == 0:
                plt.imshow(mu_states,cmap = 'hot',interpolation='nearest')
                plt.pause(0.05)

        # for s_final in gwg.states:
        #     (y,x) = gwg.coords(s_final)
        #     mu_states[y,x] = sum(mu[K,T,x,y,:])
        #
        # if np.mod(K/pltiter,pltiter) == 0:
        #     plt.imshow(mu_states,cmap = 'hot',interpolation='nearest')
        #     plt.pause(0.05)

        # Backward path
        print 'Backward path:'
        psi_states = np.full((gwg.nrows,gwg.ncols),0.0)
        for t in range(T)[::-1]:
            print 'At timestep ', t+1, '/', T
            for s_old in set(gwg.states) - set(gwg.edges):
                (y,x) = gwg.coords(s_old)

                # print 'At ', (y,x), ' cost is ', psi_states[y,x]
                for u_old in range(gwg.nactions):
                    for u in range(gwg.nactions):
                        rho[K,t,x,y,u_old,u] = np.min([rho_max,beta*cost[u] + beta*(np.abs(x-targcoords[0]) + np.abs(y-targcoords[1]))+
                                                      sum([mdp.prob_delta(s_old,u,s_new)*
                                                        psi[K,t+1,gwg.coords(s_new)[1],gwg.coords(s_new)[0],u]
                                                        for s_new in mdp.post(s_old,u)])])
                    psi[K,t,x,y,u_old] = -safe_ln(sum([nu[K,t,u_old,u]*np.exp(-rho[K,t,x,y,u_old,u])
                                                    for u in range(gwg.nactions)]),psi_max)
                    # print psi[K,t,x,y,u_old]
                    for u in range(gwg.nactions):
                        q[K,t,x,y,u_old,u] = nu[K,t,u_old,u]*safe_exp(-rho[K,t,x,y,u_old,u] + psi[K,t,x,y,u_old],0.25)
                        if (q[K,t,x,y,u_old,u] > 0 and q[K,t,x,y,u_old,u] < np.finfo(float).eps):
                            q[K,t,x,y,u_old,u] = 0

                    q[K,t,x,y,u_old,:] = q[K,t,x,y,u_old,:]/sum(q[K,t,x,y,u_old,:])

                psi_states[y,x] = sum(psi[K,t,x,y,:])
            if plot and np.mod(K/pltiter,pltiter) == 0:
                plt.imshow(psi_states,cmap = 'hot',interpolation='nearest')
                plt.pause(0.05)


def calc_mu_partial(K,t,mu,mu_states,q,qinit,exp_state,gwg,mdp):
    stateind = ['y','x']
    expind = stateind.index(exp_state)
    chpind = 1-expind
    for s_new in set(gwg.states):# - set(gwg.edges):
        exps = gwg.coords(s_new)[expind]
        chps = gwg.coords(s_new)[chpind]
        for u in range(gwg.nactions):
            if K == 0:
                mu[K,t+1,exps,chps,u] = sum([mdp.prob_delta(s_old,u_old,s_new)*
                                       qinit[gwg.coords(s_old)[expind],gwg.coords(s_old)[chpind],u]*
                                       mu[K,t,gwg.coords(s_old)[expind],gwg.coords(s_old)[chpind],u_old]
                                       for (s_old,u_old) in mdp.pre(s_new)])
            else:
                mu[K,t+1,exps,chps,u] = sum([mdp.prob_delta(s_old,u_old,s_new)*
                                       q[K-1,t,gwg.coords(s_old)[expind],gwg.coords(s_old)[chpind],u_old,u]*
                                       mu[K,t,gwg.coords(s_old)[expind],gwg.coords(s_old)[chpind],u_old]
                                       for (s_old,u_old) in mdp.pre(s_new)])
        if exp_state == 'y':
            mu_states[exps,chps] = sum(mu[K,t,exps,chps,:])
        else:
            mu_states[chps,exps] = sum(mu[K,t,exps,chps,:])
    return mu,mu_states

def calc_nu_partial(K,t,mu,nu,q,qinit,exp_state,gwg):
    if exp_state == 'y':
        exp_statespace = range(gwg.nrows)
        cheap_statespace = range(gwg.ncols)
    elif exp_state == 'x':
        exp_statespace = range(gwg.ncols)
        cheap_statespace = range(gwg.nrows)
    for chps in cheap_statespace:
        for u_old in range(gwg.nactions):
            for u in range(gwg.nactions):
                if K == 0:
                    nu[K,t,chps,u_old,u] = sum([qinit[exps,chps,u]*
                                        safe_div(mu[K,t,exps,chps,u_old],
                                                sum([mu[K,t,exps2,chps,u_old]
                                                for exps2 in exp_statespace]),0.1)
                                        for exps in exp_statespace])
                else:
                    nu[K,t,chps,u_old,u] = sum([q[K-1,t,exps,chps,u_old,u]*
                                            safe_div(mu[K,t,exps,chps,u_old],
                                                sum([mu[K,t,exps2,chps,u_old]
                                                for exps2 in exp_statespace]),0.1)
                                        for exps in exp_statespace])
                # if np.isnan(nu[K,t,chps,u_old,u]):
                    # asdf = 1
            nu[K,t,chps,u_old,:] = nu[K,t,chps,u_old,:]/sum(nu[K,t,chps,u_old,:])
    return nu

def alg_m0n1_partial(gwg,mdp,iter,T,beta,cost,exp_state,plot,pltiter):
    # Initialize parameters
    stateind = ['y','x']
    expind = stateind.index(exp_state)
    chpind = 1-expind
    if exp_state == 'y':
        exp_statespace = range(gwg.nrows)
        cheap_statespace = range(gwg.ncols)
    elif exp_state == 'x':
        exp_statespace = range(gwg.ncols)
        cheap_statespace = range(gwg.nrows)
    psi_max = beta*10000
    rho_max = beta*10000
    initcoords = tuple(reversed(gwg.coords(gwg.current[0])))
    targcoords = tuple(reversed(gwg.coords(gwg.targets[0][0])))
    # Define variables
    mu = np.full((iter,T+1,len(exp_statespace),len(cheap_statespace),gwg.nactions),0.0)
    nu = np.full((iter,T,len(cheap_statespace),gwg.nactions,gwg.nactions),0.0)
    rho = np.full((iter,T,len(exp_statespace),len(cheap_statespace),gwg.nactions,gwg.nactions),0.0)
    psi = np.full((iter,T+1, len(exp_statespace),len(cheap_statespace),gwg.nactions),0.0)  #-log(phi)
    q = np.full((iter,T,len(exp_statespace),len(cheap_statespace),gwg.nactions,gwg.nactions),0.0)

    # Initialize variables
    qinit = np.full((len(exp_statespace),len(cheap_statespace),gwg.nactions),1.0/gwg.nactions)
    for K in range(iter):
        for u in range(gwg.nactions):
            mu[K,0,initcoords[expind],initcoords[chpind],u] = 1.0/(gwg.nactions) #initial distribution
        for s in set(gwg.states):
            if gwg.coords(s) != tuple(reversed(targcoords)):
                psi[K,T,gwg.coords(s)[expind],gwg.coords(s)[chpind],:] = np.full(gwg.nactions,100) # Terminal cost

    mdp._prepare_post_cache()
    mdp._prepare_pre_cache()


    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # im = ax.imshow(np.full((gwg.nrows,gwg.ncols),0),cmap = 'hot',interpolation='nearest')
    # plt.show(block=False)
    plt.ion()
    for K in range(iter):
        print 'At iteration ', K+1, '/', iter
        # Forward path
        print 'Forward path:'
        mu_states = np.full((gwg.nrows,gwg.ncols),0.0)
        for t in range(T):
            print 'At timestep ', t+1, '/', T
            mu,mu_states = calc_mu_partial(K,t,mu,mu_states,q,qinit,exp_state,gwg,mdp) # Calculate mu
            nu = calc_nu_partial(K,t,mu,nu,q,qinit,exp_state,gwg) # Calculate nu


            if plot and np.mod(K/pltiter,pltiter) == 0:
                plt.imshow(mu_states,cmap = 'hot',interpolation='nearest')
                plt.pause(0.05)

        # for s_final in gwg.states:
        #     (y,x) = gwg.coords(s_final)
        #     mu_states[y,x] = sum(mu[K,T,x,y,:])
        #
        # if np.mod(K/pltiter,pltiter) == 0:
        #     plt.imshow(mu_states,cmap = 'hot',interpolation='nearest')
        #     plt.pause(0.05)

        # Backward path
        print 'Backward path:'
        psi_states = np.full((gwg.nrows,gwg.ncols),0.0)
        for t in range(T)[::-1]:
            print 'At timestep ', t+1, '/', T
            for s_old in set(gwg.states): # - set(gwg.edges):
                exps = gwg.coords(s_old)[expind]
                chps = gwg.coords(s_old)[chpind]
                # print (exps,chps)

                for u_old in range(gwg.nactions):
                    for u in range(gwg.nactions):
                        rho[K,t,exps,chps,u_old,u] = np.min([rho_max,beta*cost[u] + beta*(np.abs(exps-targcoords[expind]) + np.abs(chps-targcoords[chpind]))+
                                                                            sum([mdp.prob_delta(s_old,u,s_new)*
                                                                            psi[K,t+1,gwg.coords(s_new)[expind],gwg.coords(s_new)[chpind],u]
                                                                            for s_new in mdp.post(s_old,u)])])
                    psi[K,t,exps,chps,u_old] = -safe_ln(sum([nu[K,t,chps,u_old,u]*np.exp(-rho[K,t,exps,chps,u_old,u])
                                                    for u in range(gwg.nactions)]),psi_max)
                    if np.isnan(psi[K,t,exps,chps,u_old]):
                        asdf = 1
                    # print psi[K,t,exps,chps,u_old]
                    for u in range(gwg.nactions):
                        q[K,t,exps,chps,u_old,u] = nu[K,t,chps,u_old,u]*safe_exp(-rho[K,t,exps,chps,u_old,u] + psi[K,t,exps,chps,u_old],0.25)
                        if (q[K,t,exps,chps,u_old,u] > 0 and q[K,t,exps,chps,u_old,u] < np.finfo(float).eps):
                            q[K,t,exps,chps,u_old,u] = 0

                    q[K,t,exps,chps,u_old,:] = q[K,t,exps,chps,u_old,:]/sum(q[K,t,exps,chps,u_old,:])
                if exp_state == 'y':
                    psi_states[exps,chps] = sum(psi[K,t,exps,chps,:])
                else:
                    psi_states[chps,exps] = sum(psi[K,t,exps,chps,:])

            if plot and np.mod(K/pltiter,pltiter) == 0:
                plt.imshow(psi_states,cmap = 'hot',interpolation='nearest')
                plt.pause(0.05)
