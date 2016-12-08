#coding: utf-8

import numpy as np


def loadMovieLens(path='movielens'):

    # Get movie titles
    movies={}
    for line in open(path+'/u.item'):
        (id,title)=line.split('|')[0:2]
        movies[id]=title

    # Load data
    M = np.zeros( (943,len(movies.keys()))  )
    prefs={}
    for line in open(path+'/u.data'):
        (user,movieid,rating,ts)=line.split('\t')
        prefs.setdefault(user,{})
        prefs[user][movies[movieid]]=float(rating)
        
        M[int(user)-1,int(movieid)-1]=float(rating)
        
    print("Done")
    return M


def factorizeMatrix(M,K,cardIte=200,learningRate=1e-3,regularization=1e-2):
    cardUsers = np.size(M,0)
    cardMovies = np.size(M,1)

    print("cardUsers",cardUsers)
    print("cardMovies",cardMovies)
    
    P = np.random.rand(cardUsers,K)
    Q = np.random.rand(K,cardMovies)


    for ite in range(cardIte):
        for i in range(cardUsers):
            for j in range(cardMovies):
                if M[i,j]!=0:
                    derivaTemp = M[i,j] - np.inner(P[i,:],Q[:,j])

                    for k in range(K):
                        P[i,k] += learningRate*(2*derivaTemp*Q[k,j] - regularization*P[i,k])
                        Q[k,j] += learningRate*(2*derivaTemp*P[i,k] - regularization*Q[k,j])
    

        err = costFunction(M,P,Q,K,regularization)
        if err<0.001:
            break
        
        print("cost after {}/{} optimization : {}".format(ite, cardIte, err))

    return P,Q

def costFunction(M,P,Q,K,regularization):
    err = 0
    for line in range(np.size(M,0)):
        for col in range(np.size(M,1)):
            if M[line,col] > 0:
                err += (M[line,col] - np.inner(P[line,:],Q[:,col]))**2

                #Adding regularization
                for k in range(K):
                    err += (regularization/2) * (P[line,k]**2 + Q[k,col]**2) 

        return err

prefs = loadMovieLens()

P,Q = factorizeMatrix(prefs,200,2000)

print(prefs)

print(np.dot(P,Q))
    
