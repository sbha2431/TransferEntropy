__author__ = 'sudab'

__author__ = 'sudab'
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns

def safe_ln(x,out):
    if x <= 0:
        print 'safe logarithm used'
        return out
    else:
        return np.log(x)

def safe_div(x,y,out):
    if y!=0:
        return x/y
    else:
        # print 'safe division used'
        return out

def safe_exp(x,out):
    if x>200:
        print 'safe exponent used'
        return out
    else:
        return np.exp(x)

def calc_mu(K,t,mu,mu_states,q,qinit,gwg,mdp,obs_mdp,obs_states,mu_obs):
    for s_new in set(gwg.states):
        (y,x) = gwg.coords(s_new)
        for obstacle_s in obs_states:
            (y2,x2) = gwg.coords(obstacle_s)
            if K == 0:
                mu[K,t+1,x,y,x2,y2] = sum([mdp.prob_delta(s_old,u,s_new)*
                                       obs_mdp.prob_delta(obstacle_s_old,u_obs,obstacle_s)*
                                       qinit[gwg.coords(s_old)[1],gwg.coords(s_old)[0],gwg.coords(obstacle_s_old)[1],gwg.coords(obstacle_s_old)[0],u]*
                                       mu[K,t,gwg.coords(s_old)[1],gwg.coords(s_old)[0],gwg.coords(obstacle_s_old)[1],gwg.coords(obstacle_s_old)[0]]
                                       for u in range(gwg.nactions)
                                       for (s_old,uq) in mdp.pre(s_new) if s_new in mdp.post(s_old,u)
                                        for (obstacle_s_old,u_obs) in obs_mdp.pre(obstacle_s)])
            else:
                mu[K,t+1,x,y,x2,y2] = sum([mdp.prob_delta(s_old,u,s_new)*
                                        obs_mdp.prob_delta(obstacle_s_old,u_obs,obstacle_s)*
                                       q[K-1,t,gwg.coords(s_old)[1],gwg.coords(s_old)[0],gwg.coords(obstacle_s_old)[1],gwg.coords(obstacle_s_old)[0],u]*
                                       mu[K,t,gwg.coords(s_old)[1],gwg.coords(s_old)[0],gwg.coords(obstacle_s_old)[1],gwg.coords(obstacle_s_old)[0]]
                                       for u in range(gwg.nactions)
                                       for (s_old,uq) in mdp.pre(s_new) if s_new in mdp.post(s_old,u)
                                       for (obstacle_s_old,u_obs) in obs_mdp.pre(obstacle_s)])
            mu_obs[y2,x2] = sum(sum(mu[K,t,:,:,x2,y2]))
        mu_states[y,x] = sum(sum(mu[K,t,x,y,:,:]))
    # mu_states = safe_div(mu_states,sum(sum(mu_states)),0.0)
    return mu,mu_states,mu_obs

def calc_nu(K,t,mu,nu,q,qinit,gwg,obstacle_states):
    for s_old in set(gwg.states):
        (y,x) = gwg.coords(s_old)
        for u in range(gwg.nactions):
            if K == 0:
                nu[K,t,x,y,u] = sum([qinit[x,y,gwg.coords(obstacle_s_old)[1],gwg.coords(obstacle_s_old)[0],u]*
                                    safe_div(mu[K,t,x,y,gwg.coords(obstacle_s_old)[1],gwg.coords(obstacle_s_old)[0]],
                                            sum([mu[K,t,x,y,gwg.coords(obstacle_s_old2)[1],gwg.coords(obstacle_s_old2)[0]]
                                            for obstacle_s_old2 in obstacle_states]),0.1)
                                    for obstacle_s_old in obstacle_states])
            else:
                nu[K,t,x,y,u] = sum([q[K-1,t,x,y,gwg.coords(obstacle_s_old)[1],gwg.coords(obstacle_s_old)[0],u]*
                                        safe_div(mu[K,t,x,y,gwg.coords(obstacle_s_old)[1],gwg.coords(obstacle_s_old)[0]],
                                            sum([mu[K,t,x,y,gwg.coords(obstacle_s_old2)[1],gwg.coords(obstacle_s_old2)[0]]
                                            for obstacle_s_old2 in obstacle_states]),0.1)
                                        for obstacle_s_old in obstacle_states])
        nu[K,t,x,y,:] = nu[K,t,x,y,:]/sum(nu[K,t,x,y,:])
    return nu


def alg_m0n0_moveobstacle(gwg,mdp,obs_mdp,obs_states,iter,T,beta,cost,moveobstacles, plot,pltiter):
    # Initialize parameters
    psi_max = beta*100000
    rho_max = beta*100000
    initcoords = tuple(reversed(gwg.coords(gwg.current[0])))
    targcoords = tuple(reversed(gwg.coords(gwg.targets[0][0])))
    # Define variables
    mu = np.full((iter,T+1,gwg.ncols,gwg.nrows,gwg.ncols,gwg.nrows),0.0)
    nu = np.full((iter,T,gwg.ncols,gwg.nrows,gwg.nactions),0.0)
    rho = np.full((iter,T,gwg.ncols,gwg.nrows,gwg.ncols,gwg.nrows,gwg.nactions),0.0)
    psi = np.full((iter,T+1, gwg.ncols,gwg.nrows,gwg.ncols,gwg.nrows),0.0)  #-log(phi)
    q = np.full((iter,T,gwg.ncols,gwg.nrows,gwg.ncols,gwg.nrows,gwg.nactions),0.0)
    labels = np.chararray((gwg.nrows,gwg.ncols))
    for s in gwg.obstacles:
        (y, x) = gwg.coords(s)
        labels[y,x] = 'O'
    for s in gwg.targets[0]:
        (y, x) = gwg.coords(s)
        labels[y,x] = 'G'


    # Initialize variables
    qinit = np.full((gwg.ncols,gwg.nrows,gwg.ncols,gwg.nrows,gwg.nactions),1.0/gwg.nactions)
    for K in range(iter):
        mu[K,0,initcoords[0],initcoords[1],gwg.coords(moveobstacles[0])[1],gwg.coords(moveobstacles[0])[0]] = 1.0 #initial distribution
        for s in gwg.states:
            if tuple(reversed(gwg.coords(s))) != targcoords:
                psi[K,T,gwg.coords(s)[1],gwg.coords(s)[0],:,:] = np.full((gwg.ncols,gwg.nrows),10) # Terminal cost
        for t in range(T):
            for s in obs_states:
                for s2 in obs_states:
                    if s == s2:
                        psi[K,t,gwg.coords(s)[1],gwg.coords(s)[0],gwg.coords(s2)[1],gwg.coords(s2)[0]] = 150 # collision cost

    mdp._prepare_post_cache()
    mdp._prepare_pre_cache()
    obs_mdp._prepare_pre_cache()
    obs_mdp._prepare_post_cache()

    plt.ion()
    # fix,axs = plt.subplots(2,2,figsize = (10,10))
    for K in range(iter):
        print 'At iteration ', K+1, '/', iter
        # Forward path
        print 'Forward path:'
        mu_states = np.full((gwg.nrows,gwg.ncols),0.0)
        mu_obs = np.full((gwg.nrows,gwg.ncols),0.0)
        for t in range(T):
            if np.mod(t,5) == 0:
                print 'At timestep ', t+1, '/', T
            mu,mu_states,mu_obs = calc_mu(K,t,mu,mu_states,q,qinit,gwg,mdp,obs_mdp,obs_states,mu_obs) # Calculate mu
            nu = calc_nu(K,t,mu,nu,q,qinit,gwg,obs_states) # Calculate nu


            if plot and np.mod(K,pltiter) == 0:
                axcb=plt.subplot(222)
                ax = plt.subplot(221)
                ax = sns.heatmap(mu_states, vmax=1.0,vmin=0.0, annot=False, linewidths=1,linecolor='black',ax=ax, cbar=True,cbar_ax=axcb, cmap="YlGnBu")
                ax.set_title('t = {}'.format(t))
                # plt.subplot(221)
                # plt.imshow(mu_states,cmap = 'hot',interpolation='nearest')
                # plt.subplot(222)
                # plt.colorbar()
                # plt.imshow(mu_obs,cmap = 'hot',interpolation='nearest')
                plt.pause(0.05)

        for s_final in gwg.states:
            (y,x) = gwg.coords(s_final)
            mu_states[y,x] = sum(sum(mu[K,T,x,y,:,:]))

        # if np.mod(K,pltiter) == 0:
        #     plt.imshow(mu_states,cmap = 'hot',interpolation='nearest')
        #     plt.pause(0.05)

        # Backward path
        print 'Backward path:'
        psi_states = np.full((gwg.nrows,gwg.ncols),0.0)
        for t in range(T)[::-1]:
            if np.mod(t,5) == 0:
                print 'At timestep ', t+1, '/', T
            for s_old in set(gwg.states):
                (y,x) = gwg.coords(s_old)
                for obs_s in set(gwg.states):
                    (y2,x2) = gwg.coords(obs_s)
                    # print 'At ', (y,x), ' cost is ', psi_states[y,x]
                    for u in range(gwg.nactions):
                        rho[K,t,x,y,x2,y2,u] = np.min([rho_max,beta*cost[x,y,x2,y2,u] + #beta*(np.abs(x-targcoords[0]) + np.abs(y-targcoords[1]))+
                                                      sum([mdp.prob_delta(s_old,u,s_new)*obs_mdp.prob_delta(obs_s,0,obs_s_new)*
                                                        psi[K,t+1,gwg.coords(s_new)[1],gwg.coords(s_new)[0],gwg.coords(obs_s_new)[1],gwg.coords(obs_s_new)[0]]
                                                        for s_new in mdp.post(s_old,u) for obs_s_new in obs_mdp.post(obs_s,0)])])
                        if rho[K,t,x,y,x2,y2,u] == rho_max:
                            print s_old,obs_s

                    psi[K,t,x,y,x2,y2] = -safe_ln(sum([nu[K,t,x,y,u]*np.exp(-rho[K,t,x,y,x2,y2,u])
                                                    for u in range(gwg.nactions)]),psi_max)
                    # print psi[K,t,x,y,u_old]
                    for u in range(gwg.nactions):
                        # print -rho[K,t,x,y,x2,y2,u_old,u] + psi[K,t,x,y,x2,y2,u_old]
                        q[K,t,x,y,x2,y2,u] = nu[K,t,x,y,u]*safe_exp(-rho[K,t,x,y,x2,y2,u] + psi[K,t,x,y,x2,y2],0.25)
                        if (q[K,t,x,y,x2,y2,u] > 0 and q[K,t,x,y,x2,y2,u] < np.finfo(float).eps):
                            q[K,t,x,y,x2,y2,u] = 0

                    q[K,t,x,y,x2,y2,:] = q[K,t,x,y,x2,y2,:]/sum(q[K,t,x,y,x2,y2,:])

                psi_states[y,x] = sum(sum(psi[K,t,x,y,:,:]))
                # print 'cost at ', psi_states[7,3]
                # print psi_states[2,3]
            if plot and np.mod(K,pltiter) == 0:
                axcb = plt.subplot(224)
                ax = plt.subplot(223)
                sns.heatmap(psi_states,linewidths=1,linecolor='white',cbar=True,ax=ax,cbar_ax=axcb, cmap="YlGnBu")
                # plt.imshow(psi_states,cmap = 'hot',interpolation='nearest')
                plt.pause(0.05)
