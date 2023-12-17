###                             FINAL PROJECT :   CODE                                ###


import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import json 
import seaborn as sns 




# Converting files into pd dfs
artists_df = pd.read_csv('artists.dat', delimiter='\t', encoding='utf-8')
user_artists_df = pd.read_csv('user_artists.dat', delimiter='\t', encoding='utf-8')
user_tag_artists_df = pd.read_csv('user_taggedartists.dat', delimiter='\t', encoding='utf-8')
user_tag_artists_ts_df = pd.read_csv('user_taggedartists-timestamps.dat', delimiter='\t', encoding='utf-8')
tags_df = pd.read_csv('tags.dat', delimiter='\t', encoding='latin-1')
user_friend_df = pd.read_csv('user_friends.dat', delimiter='\t', encoding='utf-8')


###                                     Q1: Understanding the data-Exploration                                   ###


print("\n Q1 - Data description \n")

print('\n Briefly describe the original data set: \n')


# Making a list with our dfs, we loop the list in order to descride them all
lists_df=[artists_df,user_artists_df,user_tag_artists_df,user_tag_artists_ts_df,tags_df,user_friend_df] 
for i in lists_df:
    print(i.info())
    print('\n\n')
    print(i.describe())


# Taking the artists' name from artists_df to user_artists_df in order to show it in the frequency plot
plot1_df = pd.merge(user_artists_df, artists_df[['id', 'name']], left_on = 'artistID', right_on = 'id')

# Droppping duplicate column (id = artistID) 
plot1_df.drop('id', axis=1, inplace=True)


###                                     Frequency plot of the listening frequency of artists by users                                ###


# Creating a new df displaying the listening time of each artist in descending order
freq_artists_by_user = plot1_df.groupby('name')['weight'].sum().sort_values(ascending=False).reset_index()
print(freq_artists_by_user.head(10))

plt.figure(figsize=(20, 10))
# Using df.head(10) to output top 10 most listened artists
freq_artists_by_user.head(10).plot(x ='name', y = 'weight', kind='bar') 
plt.ylabel('Listening frequency')
plt.xlabel('Artists')
plt.xticks(rotation=45)
plt.title('Top 10 most listended artists by users')
plt.show


###                                     Frequency plot of the tags per user                                  ###


# Creating a new df displaying the tags each user has listened to in descending order
freq_tags_per_user = user_tag_artists_df.groupby('userID')['tagID'].count().sort_values(ascending=False)
print(freq_tags_per_user.head(10))
plt.figure(figsize=(20, 10))
# Using df.head(10) to output top 10 most used tags
freq_tags_per_user.head(10).plot(x = 'userID', y = 'tagID', kind='bar') 
plt.ylabel('Tag frequency', fontsize=18)
plt.xlabel('User ID', fontsize=18)
plt.xticks(rotation=45)
plt.title('Top 10 user tags', fontsize=20)
plt.show


###                                     Outlier Detection With Z-Score                                   ###


print('\n Outlier detection I: \n')

#Setting the threshold to 3, since 99.7% of the z-score fall under 3 σ
threshold = 3


###                                     Outlier Detection With Z-Score For Artists                                   ###


print('\n A. Z-score - Outliers among artists \n')
# Inserting a column which calculates the Z-score for each artists' listening weight
freq_artists_by_user['Z-score'] = (freq_artists_by_user['weight'] - freq_artists_by_user['weight'].mean()) / freq_artists_by_user['weight'].std()

# Creating and displaying a new df in which z-score value is above the threshold, thus identifying the outliers among the artists
artist_outlier = freq_artists_by_user[freq_artists_by_user['Z-score'].abs() > threshold]
print(artist_outlier)


###                                     Outlier Detection With Z-Score For Users                                   ###


print('\n B. Z-score - Outliers among users: \n')
#Creating a new df in which each user's listening count is displayed, using groupby
user_list_count = plot1_df.groupby('userID')['weight'].sum().sort_values(ascending=False).reset_index()

# Inserting a column which calculates the Z-score for each user's listening count
user_list_count['Z-score'] = (user_list_count['weight'] - user_list_count['weight'].mean()) / user_list_count['weight'].std()

# Creating and displaying a new df in which z-score value is above the threshold, thus identifying the outliers among the users
user_outlier = user_list_count[user_list_count['Z-score'].abs() > threshold]
print(user_outlier)


###                                     Outlier Detection With Z-Score For Tags                                   ###


print('\n C. Z-score - Outliers among tags: \n')
# Creating a new df in which count/tags are displayd in order to spot the outliers among tags
count_per_tag_0 = user_tag_artists_df.groupby('tagID')['artistID'].count().sort_values(ascending=False).reset_index()
count_per_tag_0 = count_per_tag_0.rename(columns={'artistID': 'Count / tag'})
count_per_tag = pd.merge(count_per_tag_0, tags_df[['tagID', 'tagValue']], on='tagID')

# Inserting a column which calculates the Z-score for how many times each tag has been used
count_per_tag['Z-score'] = (count_per_tag['Count / tag'] - count_per_tag['Count / tag'].mean()) / count_per_tag['Count / tag'].std()

# Creating and displaying a new df in which z-score value is above the threshold, thus identifying the outliers among the tags
tag_outlier = count_per_tag[count_per_tag['Z-score'].abs() > threshold]
print(tag_outlier)


###                                     Outlier Detection With IQR                                   ###


print('\n Do some research and find a second way to obtain outliers: \n \n The selected alternative way of obtaining outliers is IQR.')


###                                     Outlier Detection With IQR For Artists                                   ###


print('\n A. IQR - Outliers among artists \n')

# Defining the quantiles for the weight column
q1 = freq_artists_by_user['weight'].quantile(0.25)
q3 = freq_artists_by_user['weight'].quantile(0.75)
iqr = q3 - q1 

# Calculating bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Creating a new df dsiplaying the outliers among artists generated using IQR
artist_outlier_iqr = freq_artists_by_user[(freq_artists_by_user['weight'] < lower_bound) | (freq_artists_by_user['weight'] > upper_bound)]
artist_outlier_iqr = artist_outlier_iqr.drop(columns='Z-score')
print(artist_outlier_iqr)


###                                     Outlier Detection With IQR For Users                                   ###


print('\n B. IQR - Outliers among users \n')

# Defining the quantiles for the weight column
q1 = user_list_count['weight'].quantile(0.25)
q3 = user_list_count['weight'].quantile(0.75)
iqr = q3 - q1 

# Calculating bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Creating a new df dsiplaying the outliers among users generated using IQR
user_outlier_iqr = user_list_count[(user_list_count['weight'] < lower_bound) | (user_list_count['weight'] > upper_bound)]
user_outlier_iqr = user_outlier_iqr.drop(columns='Z-score')
print(user_outlier_iqr)


###                                     Outlier Detection With IQR For Tags                                   ###


print('\n C. IQR - Outliers among tags \n')

# Defining the quantiles for the weight column
q1 = count_per_tag['Count / tag'].quantile(0.25)
q3 = count_per_tag['Count / tag'].quantile(0.75)
iqr = q3 - q1 

# Calculating bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Creating a new df dsiplaying the outliers among users
tag_outlier_iqr = count_per_tag[(count_per_tag['Count / tag'] < lower_bound) | (count_per_tag['Count / tag'] > upper_bound)]
tag_outlier_iqr = tag_outlier_iqr.drop(columns='Z-score')
print(tag_outlier_iqr)


###                                     Testing whether datasets are normally distributed                                   ###


# Testing whether artist listening time is normally distributed by plotting a boxplot
sns.boxplot(y = 'weight', data=freq_artists_by_user)
plt.title('Artists listening time Boxplot')
plt.yscale('log')
plt.show()

# Testing whether user listening time is normally distributed by plotting a boxplot
sns.boxplot(y = 'weight', data=user_list_count)
plt.title('Users listening time Boxplot')
plt.yscale('log')
plt.show()

# Testing whether Count / tag is normally distributed by plotting a boxplot
sns.boxplot(y = 'Count / tag', data=count_per_tag)
plt.title('TagID Boxplot')
plt.yscale('log')
plt.show()


###                                        Q2: Similar Users                                  ###


print("\n Q2: Similar Users \n")

print('\n 1. You will need to find the similarity between all pairs of users, based on the artists they have heard and the ‘weight’ parameter. As a measure of similarity, you can use the cosine measure. \n')

# Creating a pivot table with users on the rows, artists on the columns, and their respective weights as values
user_artist_pivot = user_artists_df.pivot_table(index='userID', columns='artistID', values='weight', fill_value=0)


# Calculating cosine similarity between users by using cosine similarity function from sklearn.metrics.pairwise 
cosine_similarities = cosine_similarity(user_artist_pivot)

# Converting cosine similarity matrix into a pd df
cosine_similarities_df = pd.DataFrame(cosine_similarities, 
                                      index=user_artist_pivot.index, 
                                      columns=user_artist_pivot.index)

# Storing the results into a csv file 
cosine_similarities_df.to_csv('user-pairs-similarity.csv')

print('\n 1a. The results have been stored into a CSV file named user-pairs-similarity.csv \n')

print('\n 2. Find the k-nearest neighbours for each user (based on similarity defined as above) \n')

# Creating dictionaries in order to store the neighbours of each user
neighbours_3_dict = {}
neighbours_10_dict = {} 

# Using a for loop for each userID in the pivot table
for userID in user_artist_pivot.index:
    # Sorting cosine similarity for every user and choosing the top three neighbors
    neighbours_3 = cosine_similarities_df[userID].sort_values(ascending=False)[1:4].index.tolist()
    neighbours_3_dict[userID] = neighbours_3
    
    # Following the same pattern, but now for the top ten neighbors
    neighbours_10 = cosine_similarities_df[userID].sort_values(ascending=False)[1:11].index.tolist()
    neighbours_10_dict[userID] = neighbours_10


# Storing the results into the json file
with open('neighbors-3-users.json', 'w') as file:
    json.dump(neighbours_3_dict, file)

with open('neighbors-10-users.json', 'w') as file:
    json.dump(neighbours_10_dict, file)

print('\n 2a. The results have been stored into a JSON file named neighbors-k-users.json \n')


###                                     Q3: Dynamics of Listening and Tagging                                   ###


###                                     The number of users, tags, and artists per interval                                  ###


print('\n Q3: Dynamics of Listening and Tagging-20% \n The tagging of artists by the users has a time stamp. Split the data into meaningful intervals, e.g. monthly or trimester.')


# /1000 is performed in order to turn milliseconds to seconds
user_tag_artists_ts_df['timestamp'] = pd.to_datetime(user_tag_artists_ts_df['timestamp'] / 1000, unit = 's')

# Setting as the index the column timestamp
user_tag_artists_ts_df.set_index('timestamp', inplace = True)

# Creating a new pd df which displays the unique number of users, tags & artists per interval (chosen interval is monthly - 'M') using aggregation function 'nunique'
monthly_df = user_tag_artists_ts_df.resample('M').agg({
    'userID': 'nunique',
    'artistID' : 'nunique',
    'tagID' : 'nunique'
    }).reset_index()

print(f'\n 3a. The number of users, tags, and artists per month: \n\n  {monthly_df}')


###                                     The top 5 (in terms of frequency of appearance): artists & tags per interval                                  ###


# Droppping unnecessary columns and focusing in artistID 
top_artist_temp = user_tag_artists_ts_df.drop(columns = ['userID', 'tagID'])

# Adding a column that displays datetime as month
top_artist_temp['month'] = top_artist_temp.index.to_period('M')

# Grouping by month and artistID  
top_5_artists_m = top_artist_temp.groupby(['month', 'artistID']).size().reset_index(name='count')

# Grouping by month and displaying top 5 artistID by using nlargest
top_5_artists = top_5_artists_m.groupby('month').apply(lambda x: x.nlargest(5, 'count')).reset_index(drop=True)


# Droppping unnecessary columns and focusing in tagID 
top_tag_temp = user_tag_artists_ts_df.drop(columns = ['userID', 'artistID'])

# Adding a column that displays datetime as month
top_tag_temp['month'] = top_tag_temp.index.to_period('M')

# Grouping by month and tagID 
top_5_tags_m = top_tag_temp.groupby(['month', 'tagID']).size().reset_index(name='count')

# Grouping by month and displaying top 5 tagID by using nlargest
top_5_tags = top_5_tags_m.groupby('month').apply(lambda x: x.nlargest(5, 'count')).reset_index(drop=True)

# After examining below dates we found that they contained less than 5 artists or tags per interval and thus we decided to drop them
dates_to_drop = ['1956-05', '1956-08', '1957-02', '1979-04']

# For loop in which we drop above dates for the two top 5 dfs
for i in dates_to_drop:
    top_5_artists = top_5_artists.drop(top_5_artists[top_5_artists['month'] == i].index)
    top_5_tags = top_5_tags.drop(top_5_tags[top_5_tags['month'] == i].index)


print(f'\n 3b. The top 5 artists per month: \n\n {top_5_artists} \n')

print(f'\n 3b. The top 5 tags per month: \n\n {top_5_tags} \n')


###                                     Q4: Comparing Prolific User                                   ###


print('\n Q4: Comparing prolific user detect methods \n')

print('\n A. Is there a correlation between the number of artists are listened to and the number of friends a user has? \n')

# Creating a new df in which we display the unique artists listened by each user
user_artist_count = user_tag_artists_df.groupby('userID')['artistID'].nunique().reset_index(name='unique artist count')

# Creating a new df in which we display the number of friends each user has 
user_friend_count = user_friend_df.groupby('userID')['friendID'].nunique().reset_index(name='unique friend count')

# Merging the two previously created dfs 
artist_vs_friends = pd.merge(user_friend_count, user_artist_count, on = 'userID')

# Creating a matrix from the merged df in order to test whether there is a correlation between the number of artists listended and number of friends a user has
correlation_matrix_1 = artist_vs_friends[['unique friend count', 'unique artist count']].corr()

print(correlation_matrix_1)

print('\n Since, the correlation coefficent between unique artist count & unique friend count is 0.002163, we conclude there is no coreelation between the number of artists listended and the number of friends a user has \n')



print('\n B. Is there a correlation between the total listening time of a user and the number of friends he/she has? \n')

# Creating a new df in which we display the listening time of each user
user_list_time = user_artists_df.groupby('userID')['weight'].sum().reset_index(name='user listening time')

# Merging the previously created user_friend_count df with the one created above 
list_time_vs_friends = pd.merge(user_friend_count, user_list_time, on = 'userID')

# Creating a matrix from the above merged df in order to test whether there is a correlation between the listening time and number of friends a user has
correlation_matrix_2 = list_time_vs_friends[['unique friend count', 'user listening time']].corr()

print(correlation_matrix_2)

print('\n Since, the correlation coefficient is 0.23065, we conclude that there is a weak positive correlation between total listening time and number of friends a user has \n')



