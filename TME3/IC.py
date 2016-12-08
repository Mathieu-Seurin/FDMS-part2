import numpy as np

f = open('cascades_train.txt', 'r')

ep = []
i = 0

for episode in f.readlines():
    listinf = episode.split(';')
    
    
    timeStep = {}

    for couple in listinf:
        if couple[0] != '\r':
            user, time = map(lambda x:int(float(x)), couple.split(':'))
            if time in timeStep:
                timeStep[time].append(user)
            else:
                timeStep[time] = [user]

    ep.append(timeStep)



#Apprentissage

theta = np.random.uniform(0.1,0.3,(100,100))


P = [0 for i in range(100)]

eps = 0.0001
diff = 1

while diff > eps:
    for episode in ep:
        maxT = max(episode.keys())
        for t in range(1,maxT+1):
            pass
