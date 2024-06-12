#!/usr/bin/env python
# coding: utf-8

# ![alt text](https://theme.zdassets.com/theme_assets/268930/1c43f629ec1e48323c4620d081c559184af7b036.png "Logo Deci")

# # Project: Movies Dataset Analysis
# 
# ## Table of Contents :
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis (EDA)</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction :
# 
# ### Dataset Description :
# _This data set, which includes user ratings, budgets, and revenue for 10867 movies, was gathered from the IMDb website. We will analysis the data associated with movies and attempt to identify the Correlation between several variables and find why some movies has more revenue than others._
# <ul>
#     <li>columns like cast and genres has multiple values separated by {|}.</li>
#     <li>columns for budget and revenue of movie.</li>
#     <li>columns ending with (_adj)  show the budget and revenue of the associated movie in terms of 2010 dollars, accounting for inflation over time.</li>
#     <li>The director, production company, and cast columns provide details about the film's crew.</li>
# </ul>
# 
# ### Questions for Analysis :
# <ul>
# <li>Q1 : what are 10 ranked movies  according to budget , revenue , popularity ?</li>
# <li>Q2 : Does popularity and vote_average affects the revenue ?</li>
# <li>Q3 : classify movies according to profit [High profit, normal profit, low profit].</li>
# <li>Q4 : what are top 10 movies in profit ?</li>
# <li>Q5 : what are top 10 movies in runtime ?</li>
# <li>Q6 : what are least 10 movies in runtime ?</li>
# <li>Q7 : what are top 10 actors in number of movies and genres of movies they make ?</li>
# <li>Q8 : what are top 10 actors in total of revenue ?</li>
# <li>Q9 : what are top director in vote_avarage and the number of movies they make ?</li>
# <li>Q10 : what top probuction companies in number of movies ?</li>
# <li>Q11 : What production companies are ready to fund a big movie ?</li>
# <li>Q12 : What are the most profitable companies ?</li>
# <li>Q13 : Does the number of movies produced increase over the years ?</li>
# <li>Q14 : Are the movies released in the year specific to a specific season according to the months ?</li>
# <li>Q15 : What is the total number of movies in each genre? </li> 
# <li>Q16 : What is the number of movies in the top 5 genres over the years ?</li>
# <li>Q17 : what is profits types of movies over years ?</li>
# <li>Q18 : Classify movies as successful or failed.</li>
# <li>Q19 : The number of movies over years are successful and failed.</li>
# <li>Q20 : What are the number of successful and failed films in the 20th and 21th centuries ?</li>
# </ul>

# In[72]:


# import statements for all of the packages that i willl use in the project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# <a id='wrangling'></a>
# ## Data Wrangling :
# _In this process, I will load the data as a CSV file to access it, then I will clean it to remove duplicates and Nan values and drop unused columns._

# In[73]:


# load data set as csv file 
movie_df = pd.read_csv('tmdb-movies.csv')
#show first 5 rows of data set 
movie_df.head(5)


# In[74]:


#show last 5 rows from Data set 
movie_df.tail(5)


# In[75]:


#return values representing the dimensionality of the Dataset
movie_df.shape


# #### Dataset dimensions : 
# _Dataset consist of 21 columns and 10866 rows._
# 
# _I use shape and head functions to get dimensions of Dataset._

# In[76]:


#prints information about a Dataframe including the index dtype and columns, non-null values and rows .
movie_df.info()


# In[77]:


#Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution and standard deviation of quantitive Data . 
movie_df.describe()


# In[78]:


#print the descriptive data but for catergorical data including count and top frequenced 
movie_df.describe(include='object')


# ### Movies Dataset General Properties : 
# <ul>
#     <li>The Dataset consist of 21 columns and 10866 rows.</li>
#     <li>Ten  columns are quantitive Data like budget and revenue.</li>
#     <li>Eleven columns are catergorical Data like cast and director.</li>
#     <li>There is columns contian missing values like production_company column.</li>
#     <li>There is columns contian zero values (outliers) like budget column.</li>
#     <li>Release_Date column its Data type is wrong.</li>
# </ul>

# ### Data Cleaning :
# _In this process, I will clean data and prepare it for analysis._
# <ul>
#     <li>First : I will remove unused columns.</li>
#     <li>Second : I willl remove duplicate Rows.</li>
#     <li>Third : Change data type of release_date column to (datetime).</li>
#     <li>Fourth : Handling Nan values.</li>
#     <li>Fifth : Handling Outliers.</li>
# </ul>

# #### Removing Unused Columns :
# _Removing these columns:_
# 
# <ul>
#     <li>id column.</li>
#     <li>imdb_id	column.</li>
#     <li>homepage column.</li>
# </ul>
# 
# _I will remove this columns by using drop function ._

# In[79]:


#drop these columns from dataset
movie_df.drop(['id', 'imdb_id', 'homepage'], axis = 1, inplace = True)


# In[80]:


#show first 2 rows
movie_df.head(2)


# #### Removing Duplicates :
# _I'll see if there are any duplicate rows in the dataset._
# 
# _then I'll remove it if it  has duplicates._

# In[81]:


#check if dataset contain duplicate rows
movie_df.duplicated().sum()


# **_After checking if dataset contains duplicated rows or not ?_**
# 
# _The df contains 1 duplicated rows._
# 
# _I will remove it by using drop_duplicates function._

# In[82]:


#removing duplicate rows
movie_df.drop_duplicates(inplace=True)


# In[83]:


#dataset after removing duplicates
movie_df.duplicated().sum()


# ####  Convert data type of Release_date column : 
# _I notice that the data type of release_date column is (string) but it should be (datetime)._
# 
# _I will make type casting for release_date column to (datetime)._

# In[84]:


#the data type of release_date is string.
movie_df.info()


# In[86]:


#change the data type of release_date.
movie_df['release_date'] = pd.to_datetime(movie_df['release_date'])


# In[87]:


#data type after casting
movie_df.info()


# #### Handling Nan Values : 
# _we will check if dataset contain missing values (Nan) ._

# In[88]:


#check if dataset contains Nan  or not 
movie_df.isnull().sum()


# In[89]:


#Nan values in precentage by dividing it by number of rows minus one(10865) multiply * 100 .
Nan_list_precentage = (movie_df.isnull().sum()/10865)*100
Nan_list_precentage


# 
# **_6 columns contains Nan values._**
# <ul>
#     <li>cast column contain 76 Nan values with precentage 0.699494%</li>
#     <li>director column contain 44 Nan values with precentage 0.404970%</li>
#     <li>tagline column contain 2824 Nan values with precentage 25.991717%</li>
#     <li>overview column contain 4 Nan values with precentage 0.036815%</li>
#     <li>genres column contain 23 Nan values with precentage 0.211689%</li>
#     <li>production_companies contain column 1030 Nan values with precentage 9.479982%</li>
# </ul>
# 
# **_I use isnull function to get how many missing values in data set and get their sum._**
# **_I put the sum of nan values in list and divide the number of mising values  of each column to the number of rows and multiply it by 100 to get precentage_**

# In[90]:


#heatmap visualization for the distribution of Nan values over the dataset
Nan = movie_df.isnull()
plt.style.use("dark_background")
sns.heatmap(Nan)
plt.show()


# 
# _This visualization shows the distribution of missing values over dataset and their ranges in rows._
# 
# _for example in tagline column the the heatmap the missing values appear over the column._
# 
# **_After see the Missing values , their precentage and i notice that all Nan are categorical data._**
# 
# <ul>
#     <li>First : </li>
#           In tagline column the precentage of missing values to all data set is 25% so i decided to drop all column instead drop rows.
#     
#           In keywords colums the precentag of missing values to all data set is 13.7% so i decided to drop all column instead drop rows.
#           
#           I decided to drop these columns becaues their precentage of Nan and to save data as much as possible.
#           
#           I didn't drop rows because i will lose a lot of data and I don't need this columns in my analysis.
# </ul>
# <ul>
#     <li>Second : </li>
#     
#         In production_companies column the precentage of missing values to all data is 10%.
# 
#         I can't drop this column because i need it in my analysis.
#         
#         so I decided to fill the rows of missing values with unknown.
# 
# </ul>
# <ul>
#     <li>Third : </li>
#     
#         In director,cast,overview and genres columns their precentage of missing values to all data is less than 1%.
# 
#         so I decided to drop the rows of missing values.
# 
# </ul>

# In[91]:


#removing tagline and keywords columns
movie_df.drop(['tagline','keywords'], axis = 1, inplace = True)


# In[92]:


#fill missing values in production_companies column
movie_df['production_companies'].fillna('Unknown',inplace=True)


# In[93]:


#sum of Nan in all dataset
movie_df.isnull().sum().sum()


# In[94]:


#removing missing data .
movie_df.dropna(axis=0, how='any',inplace=True)


# In[95]:


#Data set after handling Nan values
movie_df.info()


# #### Handling outliers

# In[96]:


#desribe for some information about data
movie_df.describe()


# **_First : I will make boxplot to find it contians oultiers or not._**
# 

# In[97]:


#this is function make boxplot using seaborn and matplotlib i give it column name 
def boxplot_outlier(column):
    plt.boxplot(movie_df[column])
    plt.title(f'Boxplot of {column} column'.format())
    plt.ylabel('Values')
    plt.show()


# **_popularity , vote average and vote count columns :_**
# 

# In[98]:


#boxplot for popularity  column 
boxplot_outlier('popularity')


# _In popularity of some movies i notice that some movies are more popular than other more than 25._
# 
# _But most of popularity of movies between 0 and 15._

# In[99]:


#boxplot for vote_count column 
boxplot_outlier('vote_count')
#boxplot for vote_average column 
boxplot_outlier('vote_average')


# _In vote_count and average there is outliers but i will leave it because the number of people votes diff from movie to other._
# 
# **_budget, budget_adj, revenue and revnue columns :_**
# 

# In[100]:


#boxplot for revenue column 
boxplot_outlier('revenue')
#boxplot for revenue_adj column 
boxplot_outlier('revenue_adj')


# **_In revenue and revenue_dj :_**
# 
# _There some  movies didn't succeed then there revenue is zero and some movies are documentary and there is some movies there revenue is missing due to human error so I will leave outliers as it is and revenue_adj depend on revenue so there are same._

# In[101]:


#boxplot for budget and budget_adj column 
boxplot_outlier('budget')
boxplot_outlier('budget_adj')


# _In budget there is outliers and some movies has zero budget._
# 
# _So i will fill zeroes values with median because median doesn't affected by outliers and it isn't make sense that the budget of movies is 0 this may be human error so i will fill 0 by median._
# 

# In[103]:


#get zoreos
zeroes_budget = movie_df['budget']==0
movie_df.loc[zeroes_budget, 'budget'] = movie_df['budget'].mean()
#fill it with mean
zeroes_budget_adj = movie_df['budget_adj']==0
movie_df.loc[zeroes_budget_adj,'budget_adj']=movie_df['budget_adj'].mean()
boxplot_outlier('budget')
boxplot_outlier('budget_adj')


# _I fill zeroes in budget with mean and i know that mean is affected by outliers but I tried to use median but it fails becaues the number of zeroes is 5000 and median is 0 so if fill it with mean and budget_adj is depend on budget._
# 

# In[104]:


#boxplot for runtime column 
boxplot_outlier('runtime')


# **_In runtime column :_**
# 
# _There is some movies have outliers there runtime is 0 and it must at least more than 0 not zero.
# Some movie there runtime is more than 180 there are outlier but i will leave them.
# So i will fill these zeroes with median._

# In[105]:


#here i fill zeroes wiht median
zeroes_runtime = movie_df['runtime']==0
movie_df.loc[zeroes_runtime,'runtime']=movie_df['runtime'].median()
boxplot_outlier('runtime')


# **_After clean data :_**
# 
# _I will add month column which determine the month when movie released to use it to get seasons of movies over year._
# 
# _I will spilt column in cast,genres and production_companies by | to list of strings._

# In[106]:


#add release month column to my df 
movie_df['release_month'] = movie_df['release_date'].dt.month


# In[107]:


#convert from string seperated by | to list of strings 
movie_df['cast'] = movie_df['cast'].str.split('|')
#convert from string seperated by | to list of strings 
movie_df['genres'] = movie_df['genres'].str.split('|')
##convert from string seperated by | to list of strings 
movie_df['production_companies'] = movie_df['production_companies'].str.split('|')


# In[108]:


#df before EDA
movie_df.info()


# In[109]:


#some histograms of dataset
movie_df.hist(figsize=(10,8));


# <a id='eda'></a>
# ## Exploratory Data Analysis (EDA) :
# 
# ### Research Question 1 (what are 10 ranked movies  according to budget , revenue , popularity ?) : 

# In[110]:


#sort df descending according to budget
sorted_df_budget=movie_df.sort_values(by='budget',ascending=False)
#after i sort movies according to budget i get first top 10 movies 
sorted_df_budget.reset_index(inplace=True)
top10_movies=sorted_df_budget.loc[:10,['original_title','budget']]
#top 10 movies in budget 
top10_movies


# In[111]:


#barchart for top 10 movies in budget 
sns.barplot(x=top10_movies['budget'],y=top10_movies['original_title'])
plt.title('Top 10 movie in budget')
plt.show()


# In[112]:


#sort df descending according to revenue
sorted_df_revenue=movie_df.sort_values(by='revenue',ascending=False)
#after i sort movies according to revenue i get first top 10 movies 
sorted_df_revenue.reset_index(inplace=True)
top10_movies=sorted_df_revenue.loc[:10,['original_title','revenue']]
#top 10 movies in budget 
top10_movies


# In[113]:


#barchart for top 10 movies in revenue
sns.barplot(x=top10_movies['revenue'],y=top10_movies['original_title'])
plt.title('Top 10 movies in revenue')
plt.show()


# **_After get top 10 movies in budget and revenue :_**
# 
# _Not all top 10 movies in budget are in top 10 in revenue so i conclude that there is other factor that affect the movies revenue._

# In[114]:


#sort df descending according to popularity
sorted_df_popoularity=movie_df.sort_values(by='popularity',ascending=False)
#after i sort movies according to popularity i get first top 10 movies 
sorted_df_popoularity.reset_index(inplace=True)
top10_movies=sorted_df_popoularity.loc[:10,['original_title','popularity']]
#top 10 movies in budget 
top10_movies


# In[115]:


#barchart for top 10 movies in popularity
sns.barplot(x=top10_movies['popularity'],y=top10_movies['original_title'])
plt.title('Most popular Movies')
plt.show()


# **_I notice that some movies of high popularity it comes from a series of movies like the hobbit ,star wars
# so when the movie is from series related to each other its popularity increase_**

# ### Research Question 2  (Does popularity and vote_average affects the revenue ?)

# In[116]:


#correlation in df
movie_df.corr(numeric_only=True)


# **_First : I checked if there is correlation between revenue and popularity and it is more than 0.5.
# I wil make scatter plot between popularity and revenue to get the correlation betweeen them and determine if popularity affect the revenue._**

# In[117]:


#  function for scatter plot using matplotlip recieve xaxis and yaxix and colour of points
def scatter_plot(xaxis,yaxis,colour):
    plt.scatter(movie_df[xaxis],movie_df[yaxis],edgecolor='red', linewidth=1, alpha=1)
    plt.title('relation between {} and {}'.format(xaxis,yaxis))
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.tight_layout()
    plt.show()


# In[118]:


#call scatter_plot function give it popularity as xaxis , revenue as yaxis and red as colour 
scatter_plot('popularity','revenue','red')


# **_From this scatter i notice that popularity affects the revenue_**
# 
# **_Second : I checked if there is correlation between revenue and vote_average and it is more than 0.
# I wil make scatter plot between average and revenue to get the correlation betweeen them and determine if vote affect the revenue._**

# In[119]:


#call scatter_plot function give it vote_average as xaxis , revenue as yaxis and green as colour 
scatter_plot('vote_average','revenue','green')


# **_From this scatter i notice that vote affects the revenue as when the votes increases the revenue of movie increases._**
# 
# **_I wil classify vote_average values to ['low', 'good','excellent'] according to vote value then i will groupby vote after classify and revenue to get more accurte distribution for my question._**

# In[121]:


#classify vote to vote grades by using cut function 
movie_df['vote_grades'] = pd.cut(x=movie_df['vote_average'], bins=[0,4,6,10],labels=['low', 'good','excellent'])
#bar plot of relation between revenue and vote_grades 
movie_df.groupby('vote_grades')['revenue'].mean().plot(kind='barh',title='The effect of the vote on revenue',xlabel='revenue')


# **_From this bar chart we notice that when the vote are low the revenue of movies decrease and when the vote is excellent the revenue is highest.
# As we see above also from popularity in scatter plot and is affect on revenue.
# So the answer of question  
# Does popularity and vote_average affects the revenue ? 
# Is Yes ._**

# ### Research Question 3  (classify movies according to profit [High profit, normal profit, low profit,No profit])
# 
# **_Fisrt : I will make column profit is the diff of revenue and budget.
# Then I will classify it to movie profit [High profit, normal profit, low profit]._**
# 
# **_The movies of high profit are movies their profit is greater than their budget Once and a half.
# The movie of normal profit their profit less high and not less than budget.
# The movie of low profit their profit less than budget.
# The movie of no profit that their profit it negative_**
# 

# In[122]:


#first create new column for profit 
movie_df['profit']=movie_df['revenue']-movie_df['budget']
#the rules of classify 
rules = [(movie_df['profit'] <0),(movie_df['profit'] > -1) & (movie_df['profit'] <movie_df['budget']),(movie_df['profit'] >= movie_df['budget']) & (movie_df['profit'] < (movie_df['budget']*1.5)),(movie_df['profit'] >= movie_df['budget']*1.5) ]
classes = ['No_profit', 'Low_profit', 'Normal_profit', 'High_profit']
#use np select funtion to make column depend on rules
movie_df['Profit_categories']=np.select(rules,classes)
#use groupby to make bar chart for total column 
movie_df.groupby('Profit_categories')['Profit_categories'].value_counts().plot(kind='barh',edgecolor='red',linewidth=2,title='Profit for Movies')


# **_Due to there more than 5000 rows with zero revenue no_profit is much in number of movies.
# The movies of high profit are not to big  and low profit and normar profit are to low i concluede that because there is a lot of zero revenue may some of them human error._**

# ### Research Question 4  (what are top 10 movies in profit ?)
# 
# **_I will creat new df for sorted in profit and get top 10 movies in it._**

# In[123]:


#create a new df for sorted by profit descending
profit_sorted = movie_df.sort_values(by='profit',ascending = False)
profit_sorted.reset_index(inplace=True)
#get top 10 movies in profit from profit df by using loc
top10_movies=profit_sorted.loc[:10,['original_title','profit']]
#bar plot for top 10 movies in plot
sns.barplot(x=top10_movies['profit'],y=top10_movies['original_title'])
plt.title('Most profitable Movies')
plt.show()


# **_From this chart : the avatar is biggest one  and the profits of movies is near to each other except avatar._**

# ### Research Question 5 (what are top 10 movies in runtime ?)
# 

# In[124]:


#create a new df for sorted by runtime descending
runtime_sorted = movie_df.sort_values(by='runtime',ascending = False)
runtime_sorted.reset_index(inplace=True)
#get top 10 movies in profit from profit df by using loc
top10_movies=runtime_sorted.loc[:10,['original_title','runtime']]
#bar plot for top 10 movies in plot
sns.barplot(x=top10_movies['runtime'],y=top10_movies['original_title'])
plt.title('Top movies in runtime')
plt.show()


# **_The higest movies in runtime are more than 400 min_**

# ### Research Question 6  (what are least 10 movies in runtime ?)

# In[125]:


#create a new df for sorted by runtime ascending
runtime_sorted = movie_df.sort_values(by='runtime')
runtime_sorted.reset_index(inplace=True)
#get least 10 movies in profit from profit df by using loc
least10_movies=runtime_sorted.loc[:10,['original_title','runtime']]
#bar plot for least 10 movies in plot
sns.barplot(x=least10_movies['runtime'],y=least10_movies['original_title'])
plt.title('Least movies in runtime')
plt.show()


# **_The most of least movies in runtime is animatiom like Minions and the least movie in rutime is more than 2 min._**

# ### Research Question 7  (what are top 10 actors in number of movies and genres of movies they make ?)
# 
# **_In this question i want to know top 10 actors in number of movies._**
# 
# **_I also want to know if success is based on one kind of movie, like a comedian, or can succeed in more than one type._**
# 
# **_First i have actors in cast column in list so i will create new df (cast_df) to keep my df as it is to use it later and explode cast to make every actor in row without list and make the same to genres in the cast_df and i will sort it descending and make bar plot for top 10._**

# In[126]:


#creat new df and explode cast column
cast_df=movie_df.explode('cast')
#explode genres column 
cast_df=cast_df.explode('genres')
#count the number of actors  recordsa nd but the in list 
actors_counts = cast_df['cast'].value_counts()
#order actors according to movies in descending order
sorted_counts = actors_counts.sort_values(ascending=False)
#sort cast df according to number of movies of actors
cast_df = cast_df.sort_values(by='cast', key=lambda X: sorted_counts[X],ascending=False)
cast_df.reset_index(inplace=True)
#get index of top 10 
top=sorted_counts.index[:10]
top10=cast_df[cast_df['cast'].isin(top)]
#use groupby function to group between actors names and number of movies and visualize it
top10.groupby('cast')['cast'].count().plot(kind='barh',title='Top actors to number of movies',xlabel='number of movies',edgecolor='green',linewidth=2)


# **_From this chart i see that all of them makes more than 125 movies and this is big number.
# I want to know if all of these movies for an actor is from one genres or a more than one._**

# In[127]:


#get the genress of movies of top actors 
genresofactors=top10.groupby('cast')['genres'].unique()
#print actors and the genres of movies the make 
print(genresofactors)


# **_The actors can make movies of different types of genres like Antonio Banderas he makes movies in action and comedy._**
# ### Research Question 8 (what are top 10 actors in total of revenue ?)
# 
# **_First I want to know does when number of movies increase  revenue increase.
# I will creat new df and arrange it descending according to reveneu and make bar chart to show top 10._**

# In[128]:


#creat new df and explode cast column
cast_df_revenue=movie_df.explode('cast')
#sort cast_df_revenue according sum of revenue of actors
cast_df_revenue = cast_df_revenue.sort_values(by='revenue',ascending=False)
cast_df_revenue.reset_index(inplace=True)
#get of top 10 
top=cast_df_revenue['cast'].head(10)
top10=cast_df_revenue[cast_df_revenue['cast'].isin(top)]
#use groupby function to group between actors names and sum of total revenue and visualize it
top10.groupby('cast')['revenue'].sum().plot(kind='barh',title='Top actors of revenue',xlabel='sum of revenue',edgecolor='purple',linewidth=2)


# **_From this chart  the higest actor in total revnue is harrison ford .
# The number of movies doesn't affect the revenue of actors._**

# ### Research Question 9 (what are top director in vote_avarage and the number of movies they make ?)

# In[129]:


#creat new df 
director_df=movie_df
#sort director_df according vote_average of director
director_df = director_df.sort_values(by='vote_average',ascending=False)
director_df.reset_index(inplace=True)
#get of top 10 
top=director_df['director'].head(10)
top10=director_df[director_df['director'].isin(top)]
#use groupby function to group between director names and  and visualize it
top10.groupby('director')['vote_average'].mean().plot(kind='barh',title='Top director of vote',xlabel='mean of vote',edgecolor='yellow',linewidth=2)


# ### Research Question 10 (what top probuction companies in number of movies ?)
# 
# **_First i have companies in production_companies column in list so i will create new df (company_df) to keep my df as it is to use it later and explode production_companies to make every company in row without list and i will sort it descending and make bar plot for top 10._**

# In[130]:


#creat new df and explode production compaines
company_df=movie_df.explode('production_companies')
#count the numner of production companies   records and but the in list 
company_count = company_df['production_companies'].value_counts()
#order compaines according to movies in descending order
sorted_counts = company_count.sort_values(ascending=False)
#sort company df according to number of movies of company
company_df = company_df.sort_values(by='production_companies', key=lambda y: sorted_counts[y],ascending=False)
company_df.reset_index(inplace=True)
#get index of top 10 
top=sorted_counts.index[:10]
top10=company_df[company_df['production_companies'].isin(top)]
#use groupby function to group between companies names and number of movies and visualize it
top10.groupby('production_companies')['production_companies'].count().plot(kind='barh',title='Top production companies to number of movies',xlabel='number of movies',edgecolor='green',linewidth=2)


# **_In chart the unknown companies have Highest number of movies they were nan and i fill it with unknown.
# And warner bros and universal companies are the most prosduced movies._**

# ### Research Question 11  (What production companies are ready to fund a big movie ?)
# 
# **_To answer this question i should know the mean budget to get top companies so i will make new df (company_df) and sort it by budget then i will use groupby to group between companies and  mean of budget._**

# In[131]:


#creat new df 
company_df=movie_df.explode('production_companies')
#sort company_df according budget of company
company_df = company_df.sort_values(by='budget',ascending=False)
company_df.reset_index(inplace=True)
#get of top 10 
top=company_df['production_companies'].head(10)
top10=company_df[company_df['production_companies'].isin(top)]
#use groupby function to group between company names and  mean of budget and visualize it
top10.groupby('production_companies')['budget'].mean().plot(kind='barh',title='Companies that are able to fund huge movies',xlabel='mean of budget',edgecolor='orange',linewidth=2)


# **_According to what companies paid to make movies we can get which companies are able to fund big movies like Boran company._**

# ### Research Question 12 (What are the most profitable companies ?)
# 
# **_To answer I will make new df (company_df) and sort it by profit then i will use groupby to group between companies and profit._**

# In[132]:


#creat new df 
company_df=movie_df.explode('production_companies')
#sort company_df according profit of company
company_df = company_df.sort_values(by='profit',ascending=False)
company_df.reset_index(inplace=True)
#get of top 10 
top=company_df['production_companies'].head(10)
top10=company_df[company_df['production_companies'].isin(top)]
#use groupby function to group between company names and  mean of budget and visualize it
top10.groupby('production_companies')['profit'].sum().plot(kind='barh',title='Most profitable companies',xlabel='Profit',edgecolor='white',linewidth=2)


# **_The most companies gains profit form their movies.
# Highest 2 companies in profit are paramount and fox film._**

# ### Research Question 13  (Does the number of movies produced increase over the years ?)
# 
# **_First i will get the total number pf movies produced per year._**

# In[133]:


#totak number of movies per year by using groupby
movies_to_years=movie_df.groupby('release_year')['release_year'].count()
#print it
print(movies_to_years)


# **_we see that the number of movies increases by year._**

# In[134]:


#plot of change of number of movies over years
plt.plot(movies_to_years,'--')
plt.title('Change of number of movies over years ')
plt.ylabel('Number of movies')
plt.xlabel('Years')
plt.show()


# **_From this chart i coclude that : The number of movies is increases every year_**
# 
# **_The number of movies form 1960 to 1070 per year are less than 100 because in this period where technology of movies and tv are just inveted so not all of people have tv and go to cinema._**
# 
# **_The number movies from 1970 to 1990 increaed slowly from less 100 to more than 100 movies per the technologies evolved and computer are Spreaded that help companies to make more movies._**
# 
# **_From 1990 till now increaed faster and became more than 700 per year is huge increase due to evolve of internet and electonic devices and people can get any movie they want to see online and marketing became global._**

# ### Research Question 14 (Are the movies released in the year specific to a specific season according to the months ?)
# 
# **_In this question i want to know if the movie is release in seasons in year to get high profit._**

# In[135]:


#count the number of movies released per months by using groupby
movies_to_months = movie_df.groupby('release_month')['release_month'].count()
#print it
print(movies_to_months)


# **_The number of movies changes in months like in 10 , 9 months is more than 1300 and other is less 1000._**

# In[136]:


#visualize to make it more clear 
movie_df.groupby('release_month')['release_month'].count().plot(kind='bar',title='',ylabel='Numbers of movies',edgecolor='gray',linewidth=3)


# **_The end of the year from 8 to 12 and in jan. have high number of movies._**
# 
# **_I wil get the total of movies to seasons of year._**

# In[137]:


#coditions to classify months to seasons
rules = [(movie_df['release_month'] <3) | (movie_df['release_month'] ==12) ,(movie_df['release_month'] >2) & (movie_df['release_month'] <6),(movie_df['release_month'] >5) & (movie_df['release_month'] <9),(movie_df['release_month'] >8) & (movie_df['release_month'] <12) ]
classes = ['Winter', 'Spring', 'Summer', 'Autumn']
#make new column for season
movie_df['seasons']=np.select(rules,classes)
#chart for number of movies to seasons
movie_df.groupby('seasons')['release_month'].count().plot(kind='barh',title='Number of movies in seasons',xlabel='Number of movies',edgecolor='pink',linewidth=5)


# **_From this charh i see that autumn has the biggest number of movies to other seasons._**
# 
# **_In these months, holidays and events such as Christmas, Halloween, vacations, encourage companies to make more movies in these periods of the year, which helps to increase profits in autumn season and 1,9 and 10 months._**

# ### Research Question 15 (What is the total number of movies in each genre ? )

# In[138]:


#create new df for genres
genres_df=movie_df.explode('genres')
#use groupby to get the number of movies in each genres and visualize it
genres_df.groupby('genres').size().plot(kind='barh',title='The number of movies in each genres',xlabel='counts',edgecolor='green',linewidth=5)


# **_Top 4 genres are Drama , Action , crime and thriller are more than 2000 movies._**
# 
# **_The other genres are less 200 movies and because people looking for top 4 genres much than other._**

# ### Research Question 16 (What is the number of movies in the top 5 genres over the years ?)
# 
# **_First : I have years from 1960 to 2015 and this is huge number of years to visulize it so first i will create new column decade and divide this years to decade._**
# 
# **_Second : I will create new df and explode genres column from list of strings to string in each row an order it descending to number of movies to each genres then i will visualize top 5 genres to see the changes of number of moives to it over decades._**

# In[139]:


#function to get decade by divide year by 10 without remined and multiply it by 10 to retern it as year again
def classifytodecade(year):
  return (year // 10) * 10
#classify years to decades by using appy and call classifytodecade func. to get decade
movie_df['decade'] = movie_df['release_year'].apply(classifytodecade)
#show top 5 rows of df after divided years to decades 
movie_df.head()


# In[140]:


#create new df and explode genres in new df
genres_df_years=movie_df.explode('genres')
#count the number of genres records and but the in list 
genres_count = genres_df_years['genres'].value_counts()
#order genres to movies in descending order
sorted_counts_genres = genres_count.sort_values(ascending=False)
#sort genres df according to number of movies of each genres
genres_df_years = genres_df_years.sort_values(by='genres', key=lambda y: sorted_counts_genres[y],ascending=False)
genres_df_years.reset_index(inplace=True)
#get index of top 10 
top5=sorted_counts_genres.index[:5]
top_5=genres_df_years[genres_df_years['genres'].isin(top5)]
#use groupby function to group between decade and top 5 genres and visualize it
top_5.groupby(['decade', 'genres']).size().unstack().plot(title='Top 5 genres over years',xlabel='decade',ylabel='Number of movies')


# **_Top 5 geners are action , comedy , drama , romance and thriller._**
# 
# **_The number of movies in each genres started io increase from 1970 to 2010 as we also see above the changes of numbrt og movies after apear internet._**

# ### Research Question 17  (what is profits types of movies over years ?)

# In[141]:


#use groupby to get the profit type of movies over decade 
movies = movie_df.groupby(['decade','Profit_categories']).size().unstack().plot(title='Profit type of Movies',xlabel='decade',ylabel='Number of movies')


# ### Research Question 18  (Classify movies as successful or failed.)
# 
# **_Conditions of movies classsify:_**
# <ul>
#     <li>First : If profit of movie more than 0 then this movie is successful.</li>
#     <li>Second : If profit of movie less than 0 and the genres of movie isn't documentary is failed.</li>
#     <li>Third : If Movie is documentary and its vote isn't low is successful.</li>
#     <li>Fourth : If Movie is documentary and its vote is failed.</li>
# </ul>

# In[142]:


#function to classify movies to conditions
def classiymovies(df):
  if df['profit'] > 0 :
      return 'successful'
  elif df['profit'] <=0 and 'Documentary' not in df['genres']:
      return 'failed'
  elif df['profit'] <=0 and 'Documentary' in df['genres'] and df['vote_grades']!='low' :
      return 'successful'
  else :
      return 'failed'
#create new column for movie classify
movie_df['Movies_status'] = movie_df.apply(classiymovies,axis=1)
#df after addding new column
movie_df.head()


# ### Research Question 19  (The number of movies over years are successful and failed.)

# In[143]:


#use groupby function to the number of movies over years are successful and failed
movie_df.groupby(['decade', 'Movies_status']).size().unstack().plot(kind='bar',title='The number of movies atatus of years',xlabel='decade',ylabel='Number of movies')


# **_The number of failed movies is less than number of successful movies, I think that becaues huge number of movies have 0 revenue._**

# ### Research Question 20  (What are the number of successful and failed movies in the 20th and 1st centuries? )
# 
# **_First : I will create new column and divide years to centuries less 2000 is 20 century and more than or equal 2000 is 21 century._**
# 
# **_Second  : I visualize number of movies over centuries is successful and failed._**

# In[144]:


#function to classify years to centuries 
def century(df):
   if df['release_year'] < 2000 : 
       return '20'
   else : 
       return '21'
#create new column for century classify
movie_df['century'] = movie_df.apply(century,axis=1)
#df after adding new column
movie_df.head()


# In[145]:


#use groupby to get relation between changes of movies in two centuries 
movie_df.groupby(['century', 'Movies_status']).size().unstack().plot(kind='bar',title='The number of movies status over centuries',xlabel='centuries',ylabel='Number of movies')


# **_We see huge different in number of movies in 20 cetury and 21 cetury almost double number of movies is increased due to evolve of technology and internet._**

# <a id='conclusions'></a>
# ## Conclusion :
# 
# ### Limitations:
# **_There is some problems in dataset was big problem to me and Obstruction for me in analysis are :_**
# <ul>
#     <li>More than half of [budget,revenue,budget_adj,revenue_adj]  columns was zeroes.</li>
#     <li>The precentage of missing values in [keywords,tagline] columns so that lead me to delete all colum without use it in my analysis.</li>
#     <li>Huge number of outliers in all quantitive columns that gives me sometimes gives me unaccurate visualizations.</li>
#     <li>More than 8 precentage of companies is missing that lead me to fill it with unknown and i lost a valuable information cloud be useful in analysis.</li>
# </ul>
# 
# 
# 
