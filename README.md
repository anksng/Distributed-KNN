# Distributed-K means

### Implementation of K Means using MPI framework

I broke down the Kmeans algorithm into four parts: 
* Initializing  the centroid and  broadcast. 
* Assigning  Membership  to the data instances (based on euclidean distance) 
* Calculating the mean of membership clusters to  update centroids 
* Converge  - when the centroids stop updating 
### Dataset 
“Twenty Newsgroups” corpus (data set) available at UCI repository https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups

### Results:
Check the ipnyb for the speed up in performance while using different number of workers.
