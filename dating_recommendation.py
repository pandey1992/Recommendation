

import numpy as np
import pandas as pd

"""Data for this dating profile is available at www.libimseti.cz - Dating website recommendation (collaborative filtering) http://www.occamslab.com/petricek/data/

Indian Matrimonial DataSet at http://iacs-courses.seas.harvard.edu/courses/iacs_projects/matrimony_data_exploration/introduction.html. You may want to look at the dataset. This dataset is huge, so many csv's.
https://github.com/deadofied/ac299r/tree/master/data
"""

columns= ['UserID','ProfileID','Rating']
df_dating = pd.read_csv('ratings.csv',names = columns)

df_dating['UserID'].nunique(), df_dating['ProfileID'].nunique()

#ttps://drive.google.com/open?id=1XJKFEpMvXJOzkcxZgncLGJoHWr7bE7m-

columns= ['Gender']
df_gender = pd.read_csv('gender.csv',names = columns)


profile_mean_rating = pd.DataFrame(df_dating.groupby('ProfileID')['Rating'].mean())

user_mean_rating =pd.DataFrame(df_dating.groupby('UserID')['Rating'].mean())

profile_mean_rating['number of rating'] = pd.DataFrame(df_dating.groupby('ProfileID')['Rating'].count())

user_mean_rating['number of rating'] = pd.DataFrame(df_dating.groupby('UserID')['Rating'].count())

user_new_meanrate = user_mean_rating[user_mean_rating['number of rating'] > 300]



user_index = user_new_meanrate.index

profile_new_meanrate = profile_mean_rating[profile_mean_rating['number of rating'] > 300 ]

profile_index = profile_new_meanrate.index


df_dating_sample = df_dating.loc[(df_dating['UserID'].isin(user_index)) & (df_dating['ProfileID'].isin(profile_index))]



profile_mean_rating = pd.DataFrame(df_dating_sample.groupby('ProfileID')['Rating'].mean())
profile_mean_rating['number of rating'] = pd.DataFrame(df_dating_sample.groupby('ProfileID')['Rating'].count())
user_mean_rating =pd.DataFrame(df_dating_sample.groupby('UserID')['Rating'].mean())
user_mean_rating['number of rating'] = pd.DataFrame(df_dating_sample.groupby('UserID')['Rating'].count())



"""# Visulaizing the Profile rating dataset to undersatnd how the profile are rated and top profiles"""

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,8))
profile_mean_rating['Rating'].hist(bins=120)

"""**In the above plot we see the graph follows a normal distribution except at the peaks ratings 1 and 10 which tells us that there are a lot of profile which are not liked by many user and lots of profiles heavily liked by users.**"""

profile_mean_rating['number of rating'].sort_values(ascending=False).head()

sns.jointplot(x='Rating',y='number of rating',data=profile_mean_rating,alpha=0.5)

"""**We can see that for profiles around 6 to 8 is rated most and also number of profiles which are rated 10 is quite large in number. This indicates that most people rate the profile they like.**

**Plan of action for this week:**

**1.Popularity based recommendation
2.Content based recommendation
3.User-User Recommendation
4.Item-Item Recommendation
5.SVD(without applying SGD to improve the weights)**
"""

#Popularity based recommendation will recommend top 10 profiles which are rated by most number of user and are greater than 9.

#Just writing the code snippet before putting the code in function
  
df_profile = profile_mean_rating.sort_values(['number of rating','Rating'],ascending=[False,False])
df_profile.head()

len(profile_mean_rating[(profile_mean_rating['number of rating'] > 10000) & (profile_mean_rating['Rating'] > 9)])

#Defining Popularity Based Model
def popular_recommend_model(userID,profile_df,threshold,k):
  """
  It will return top 'k' profiles based on highest rating and number of ratings greated than the defined threshold irrespective of User ID,
  If value of K is greater than the number of profile which matches the given criteria, then it is just going to return maximum profile 
  matching the given criteria.
  """
 # try:
#   if len(profile_df[(profile_df['number of rating'] > threshold) & (profile_df['Rating'] > 9)]) < 1:
#     print("Your filtering doesn't match any profiles")
  return (profile_df[(profile_df['number of rating'] > threshold) & (profile_df['Rating'] > 9)].head(k).index)
 # except:
 #   print("Index out of range")

popular_recommend_model(123,profile_mean_rating,2500,10)

#2. Content- Based Recommendation
#It will check the similarity of profiles based on different latent features and recommend profile similar to the that profiles based on user 
#previous profile rating
#creating Profile Matrix which will hold user as index and profile as column and their rating as values
profilemat = df_dating_sample.pivot_table(index='UserID',columns='ProfileID',values='Rating')

"""**Putting all these in a function which will take any profile ID and return most similar profile to that profile given the profile has atleast 200 ratings**"""

def content_based_recommend(profileID, profilemat,profile_mean_rating):
  """
  This function which will take any profile ID and return most similar profile to that profile from user-profile rating matrix
  given the profile has atleast 200 ratings
  """
  rating_profileID = profilemat[profileID]
  corr_profileID = pd.DataFrame(profilemat.corrwith(rating_profileID),columns=['Correlation'])
  corr_profileID.dropna(inplace=True)
  corr_profileID = corr_profileID.join(profile_mean_rating['number of rating'])
  return (corr_profileID[corr_profileID['number of rating'] > 300].sort_values('Correlation',ascending=False).head(20))

content_based_recommend(175,profilemat,profile_mean_rating)



"""For User-item or item-item memory based Collaborative Filetering, we need to make a matrix in which we'll be storing the pairwise distance;  Also do we need to split our dataset into train and test set? Because in which we'll be storing the distance right?
We'll be applying this  right?
<br> <h3> For item-item <br>
  <p> Cosine Similarity </p>
<img class="aligncenter size-thumbnail img-responsive" src="https://latex.codecogs.com/gif.latex?\hat{x}_{k,m}&space;=&space;\frac{\sum\limits_{i_b}&space;sim_i(i_m,&space;i_b)&space;(x_{k,b})&space;}{\sum\limits_{i_b}|sim_i(i_m,&space;i_b)|}"/>

User -Item Similarity : Users who are similar to you also liked

Similarity values between users are measured by observing all the items that are rated by both users.

<img class="aligncenter size-thumbnail img-responsive" src="https://latex.codecogs.com/gif.latex?\hat{x}_{k,m}&space;=&space;\bar{x}_{k}&space;&plus;&space;\frac{\sum\limits_{u_a}&space;sim_u(u_k,&space;u_a)&space;(x_{a,m}&space;-&space;\bar{x_{u_a}})}{\sum\limits_{u_a}|sim_u(u_k,&space;u_a)|}"/>
"""

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df_dating, test_size = 0.25)

n_users_all = df_dating['UserID'].unique().shape[0]
n_items_all = df_dating['ProfileID'].unique().shape[0]
n_users_all, n_items_all

n_users = df_dating_sample['UserID'].unique().shape[0]
n_items = df_dating_sample['ProfileID'].unique().shape[0]
n_users, n_items



df_dating_sample['ProfileID'].unique().shape[0]

df_dating_sample.shape, df_train.shape, df_test.shape

for line in df_dating_sample.itertuples():
  print(line)
  break

train_data_mat = np.zeros((n_users, n_items))

#Create two user-item matrix, one for training and another for testing
train_data_mat = np.zeros((n_users, n_items))
#itertuples gives you content of the row along with it's index
for line in df_train.itertuples():
  #line[1] - 1 is used to account for index to start from 0, so we get value like
  #train_data_mat[userID,ProfileID] = Rating
  train_data_mat[ line[1] -1, line[2] -1 ] = line[3]
test_data_mat = np.zeros((n_users, n_items))
for line in df_test.itertuples():

  test_data_mat[ line[1] -1, line[2] -1 ] = line[3]

"""In recommendation system, we commonly use cosine matrix which will treat the ratings given by a specific user as a vector assuming all the null values as 0 rating and then the cosine gives you the measure of similarity as we know cos(0) = 1 so a value of 1 means the users are very similar.

Cosine similiarity for users *a* and *m* can be calculated using the formula below, where you take dot product of  the user vector *$u_k$* and the user vector *$u_a$* and divide it by multiplication of the Euclidean lengths of the vectors.
<img class="aligncenter size-thumbnail img-responsive" src="https://latex.codecogs.com/gif.latex?s_u^{cos}(u_k,u_a)=\frac{u_k&space;\cdot&space;u_a&space;}{&space;\left&space;\|&space;u_k&space;\right&space;\|&space;\left&space;\|&space;u_a&space;\right&space;\|&space;}&space;=\frac{\sum&space;x_{k,m}x_{a,m}}{\sqrt{\sum&space;x_{k,m}^2\sum&space;x_{a,m}^2}}"/>

To calculate similarity between items *m* and *b* you use the formula:

<img class="aligncenter size-thumbnail img-responsive" src="https://latex.codecogs.com/gif.latex?s_u^{cos}(i_m,i_b)=\frac{i_m&space;\cdot&space;i_b&space;}{&space;\left&space;\|&space;i_m&space;\right&space;\|&space;\left&space;\|&space;i_b&space;\right&space;\|&space;}&space;=\frac{\sum&space;x_{a,m}x_{a,b}}{\sqrt{\sum&space;x_{a,m}^2\sum&space;x_{a,b}^2}}
"/>
"""





from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

"""Now we will make prediction based on the similarity of the above calculated matrix. For user based CF, we will use the below formula.

<img class="aligncenter size-thumbnail img-responsive" src="https://latex.codecogs.com/gif.latex?\hat{x}_{k,m}&space;=&space;\bar{x}_{k}&space;&plus;&space;\frac{\sum\limits_{u_a}&space;sim_u(u_k,&space;u_a)&space;(x_{a,m}&space;-&space;\bar{x_{u_a}})}{\sum\limits_{u_a}|sim_u(u_k,&space;u_a)|}"/>

We are subtracting the user mean rating from each rating which would take care of differences between user while rating. Say for example, a user m rates good profile with 8 and the profile he dislikes with 2 and similarly other user a rates movies he likes with 10 and other movies 4. Now both the users have similar taste but they rate differently. After removing the average rating of a user, we also need to normalize so the rating doesn't cross 10. For this we have divided with the 

And for item based CF, we will use the below mentioned formula:

<img class="aligncenter size-thumbnail img-responsive" src="https://latex.codecogs.com/gif.latex?\hat{x}_{k,m}&space;=&space;\frac{\sum\limits_{i_b}&space;sim_i(i_m,&space;i_b)&space;(x_{k,b})&space;}{\sum\limits_{i_b}|sim_i(i_m,&space;i_b)|}"/>
"""

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
        rate_pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        rate_pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])     
    return rate_pred

item_prediction = predict(train_data_mat, item_similarity, type='item')
user_prediction = predict(train_data_mat, user_similarity, type='user')

"""We will use RMSE to evaluate the predicted ratings.

Now for calculating RMSE , we would like to filter out the rating which re available in the test_data_mat. This could be done by using nonzero method on the series which would provide the index were the values are not zero.
"""

from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, test):
    prediction = prediction[test.nonzero()].flatten() 
    test = test[test.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, test))

print('User-Item similarity CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('Item-Item similarity CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))

"""**Model Based Collaborative Filtering**

Model based collaborative filtering is based on Matrix Factorisation which is a unsupervised learning. The goal of factorisation is to find the latent features of Users and Latent features of items.

Matrix factorization models map both users and items to a joint latent factor space of dimensionality, such that user-item interactions are modeled as inner products in that space. Accordingly, each item i is associated with avector qi ∈ train_data_mat(User-Item rating matrix), and each user u is associated with a vector pu ∈ train_data_mat(User-Item rating matrix) .

For a given item i, the elements of qi measure the extent to which the item possesses those factors, positive or negative. For a given user u, the elements of pu measure the extent of interest the user has in items that are high on the corresponding factors, again, positive or negative. The resulting dot product,
qi.T.pu
 captures the interaction between user u and item i—the user’s overall interest in the item’s characteristics. This approximates user u’s rating of item i, which is denoted by rui, leading to the estimate
rˆui = qi.T .pu.

**SVD**
A well known method for Matrix Factorisation is Singular Value Decomposition(SVD). The general equation for SVD is expressed as follows:
<img src="https://latex.codecogs.com/gif.latex?M=USV^T" title="M=USV^T" />

Given `m x n` matrix `M`:
* *`U`* is an *`(m x r)`* orthogonal matrix
* *`S`* is an *`(r x r)`* diagonal matrix with non-negative real numbers on the diagonal
* *V^T* is an *`(r x n)`* orthogonal matrix

Elements on the diagnoal in `S` are known as *singular values of `M`*. 


Matrix *`M`* can be factorized to *`U`*, *`S`* and *`V`*. The *`U`* matrix represents the feature vectors corresponding to the users in the hidden feature space and the *`V`* matrix represents the feature vectors corresponding to the items in the hidden feature space.
"""

import scipy.sparse as sp
from scipy.sparse.linalg import svds

#get SVD components from train matrix. Choose k.
u, s, vt = svds(train_data_mat, k = 50)
s_diag_matrix=np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
print('Model-based CF MSE: ' + str(rmse(X_pred, test_data_mat)))











