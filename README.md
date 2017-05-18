# Facebook-Community-Detection-and-Link-Prediction

### Here we Implement Community detection and Link prediction algorithms using Facebook's "like" data.


The data we are using is collected using snowball sampling. We have used a Test case user in our case it is "Bill Gates". We found all the people he "likes" (facebook terminology) and then for each new user we discovered, I further find the people they liked.
All these "like" data is then stored in a gzip file named "edges.txt.gz" which indicates the relationships between facebook users. 

### The Implementation is done in two phases:
#### 1. Community Detection:

After the data collection process is completed we make a graph network of nodes and edges. Now to find communities in this graph we use the Girvan Newman Approach. 
(All the algorithms are developed from scratch without using any predefined libraries so as to understand the working of each part)

#### 2. Link Prediction:

Now for the purpose of link prediction we now already have a Graph of Bill Gates "like" data. 
We remove 5 of the accounts liked by Bill Gates.
Then we use this newly created graph and do Link Prediction, finally compute the accuracy of our prediction. 
