
# coding: utf-8


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.size
from time import time

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

#vectors= []
cat = ['alt.atheism','comp.graphics']
#,'comp.os.ms-windows.misc']
#,'comp.os.ms-windows.misc']
#        , 'comp.sys.ibm.pc.hardware'] 
#        ,'comp.sys.mac.hardware',
#  'comp.windows.x',
#  'misc.forsale',
#  'rec.autos']
#  'rec.motorcycles',
#  'rec.sport.baseball']
#  'rec.sport.hockey',
#  'sci.crypt',
#  'sci.electronics',
#  'sci.med',
#  'sci.space',
#  'soc.religion.christian',
#  'talk.politics.guns',
#  'talk.politics.mideast',
#  'talk.politics.misc',
#  'talk.religion.misc']
newsgroups = fetch_20newsgroups(categories=cat)
vectorizer = TfidfVectorizer(stop_words='english')
data = vectorizer.fit_transform(newsgroups.data)
data = data.toarray()
print(data)
#np.random.seed(10)
#data = np.random.randint(5,size=(1000,500))

def calculateNewCenter(data_class):
    newCenters=[]
    for x_i in data_class[0]: 
        c=np.mean(data_class[0][x_i], axis=0)
        newCenters.append(c)
    return newCenters



k = 2

#centroids = {}


centroids= []
for j in range(k):
    #centroids[i]= data[i]
    centroids.append(data[j])

print("all centroids",centroids)
#for i in range(2,size):
#    comm.send(data[(rank-2)*len(data):(rank-1)*(len(data))],dest=i,tag = 10)
tol=0.1
t1= time()
itr=20
for i in range(itr):
    print("I am in iteration # ",i)
    centroids = comm.bcast(centroids, root=0)
    if rank == 0:
    #        print("whole data:",data)
        
        
        sendbuf = np.array_split(data,size,axis=0)
        data_class = None
        
    else:
        sendbuf= None
       
    #if rank <size:
    recbuf=comm.scatter(sendbuf, root=0) 
    
#    print("rec chunk",rank,recbuf)
    #    print(centroids)
    #    print(recbuf)
    classifications = {}
    print("centroids for me",rank,(centroids))         
    for i in range(k):
        classifications[i] = []
            
    for featureset in recbuf:
        distances= [ np.linalg.norm(featureset-centroids[centroid]) for centroid in range(k)]
                    
        classification = distances.index(min(distances))
                    
        classifications[classification].append(featureset)
#        print("i am rank :",rank,classifications)
    data_class = comm.gather(classifications,root=0)
#    print("class data",(data_class))
#    print("data whole",data)

        
    if rank ==0:
        
      
                
#        print("classes => ", classes)

        nc = calculateNewCenter(data_class)
#        print("data class",(classes))
        
        
        centroids = nc
            
            
        print("new Centers:",centroids)
            
print("Time taken :",(time()-t1))       
   
    
    
#print("final classification:",data_class)
#     
#print("calssifications:",rank,classifications)
#print("rank :",rank,centroids)
#print("rank rec dat:",recbuf)



#for i in range(max_iter):
#if rank ==1:
#    for i in range(2,size):
#        comm.send(data[(rank-2)*len(data):(rank-1)*(len(data))],dest=i,tag = 10)
#if rank >1:
#        
#        datachunk= [ ]
#        for i in range(2,size):
#            x=comm.recv(source=i,tag = 10)
#            datachunk.append(x)
#        print(datachunk)
#        classifications = {}
#                
#        for i in range(k):
#            classifications[i] = []
#            
#        for featureset in datachunk:
#            distances= [ np.linalg.norm(featureset-centroids[centroid]) for centroid in centroids]
#            
#            classification = distances.index(min(distances))
#                    
#            classifications[classification].append(featureset)
#        print("classsss",classifications)
#        x = comm.send(classifications, dest=1, tag=11)
#        print("sent dat ",x)

#if rank ==1:
#        
#    data = {}
#    for i in range(2,size):
#        data1 = comm.recv(source=i, tag=11)
##            print("data1 :",data1)
#        data.update(data1)
#        print("rec_data",data)






