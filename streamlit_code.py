%%writefile ap.py
import io
import numpy as np
import pandas as pd
credits_df= pd.read_csv('/content/credits.csv')
movies_df = pd.read_csv('/content/movies (1).csv')
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Assuming that both DataFrames have a column named 'title' for merging
movies_df = movies_df.merge(credits_df, on="title")

import pandas as pd

# Assuming you have merged movies_df and credits_df, resulting in the merged DataFrame
# You may want to adjust this depending on your actual DataFrames
merged_df = movies_df.merge(credits_df, on="title")

# Get a list of all column names
all_columns = merged_df.columns

# Create new column names without suffixes
new_columns = [column.replace('_x', '').replace('_y', '') for column in all_columns]

# Assign the new column names to the DataFrame
merged_df.columns = new_columns
#===================================================================================================================
movies_analysis= movies_df[['keywords','budget','cast','crew','title','revenue','vote_average', 'vote_count']]
# Now, merged_df should have columns without suffixes
#=======================================================================================================================
movies_df = movies_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]


movies_df.dropna(inplace=True)
import ast
import pandas as pd

# Define the conversion function
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
#----------------------------------------------------------------------------
movies_analysis['keywords'] = movies_analysis['keywords'].apply(convert)
#----------------------------------------------------------------------------
# Assuming you have a DataFrame named movies_df
# Correct column names and apply the conversion function

movies_df['genres'] = movies_df['genres'].apply(convert)
movies_df['keywords'] = movies_df['keywords'].apply(convert)
import ast
import pandas as pd

# Define the conversion function
def convert3(obj):
    L = []
    counter = 0  # Initialize the counter
    for item in ast.literal_eval(obj):
        if counter < 3:  # Limit to the first 3 elements
            L.append(item['name'])
            counter += 1
        else:
            break
    return L

# Assuming you have a DataFrame named movies_df
# Correct column names and apply the conversion function
#------------------------------------------------------------------------------
movies_analysis['cast'] = movies_analysis['cast'].apply(convert3)
#---------------------------------------------------------------------------------
movies_df['cast'] = movies_df['cast'].apply(convert3)
import ast
import pandas as pd

# Define the conversion function
def fetch_director(obj):
    director_list = []  # Initialize an empty list to store director names
    for item in ast.literal_eval(obj):
        if item['job'] == 'Director':
            director_list.append(item['name'])
            break  # Exit the loop once the director is found
    return director_list
#----------------------------------------------------------------------------------------------------
def crew_l(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
movies_analysis['crew']=movies_analysis['crew'].apply(crew_l)
movies_analysis['crew'] = movies_analysis['crew'].apply(lambda x: list(set(x)))
movies_analysis['keywords'] = movies_analysis['keywords'].apply(lambda x: [i.replace("£", "") for i in x])
movies_analysis['cast'] = movies_analysis['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies_analysis['crew'] = movies_analysis['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
movies_analysis['tags'] =movies_analysis['keywords'] + movies_analysis['cast'] + movies_analysis['crew']
movies_analysis['tags'] = movies_analysis['tags'].apply(lambda x: ' '.join(x))
movies_analysis['tags'] = movies_analysis['tags'].apply(lambda x: x.lower())
# Concatenate the columns and store the result in the 'tags' column
#--------------------------------------------------------------------------------------------------
# Assuming you have a DataFrame named movies_df
# Correct column names and apply the conversion function

movies_df['crew'] = movies_df['crew'].apply(fetch_director)
movies_df['overview'] = movies_df['overview'].apply(lambda x: x.split())
# Fixing the code for genres, keywords, and cast columns
movies_df['genres'] = movies_df['genres'].apply(lambda x: [i.replace("*", "") for i in x])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x: [i.replace("£", "") for i in x])
movies_df['cast'] = movies_df['cast'].apply(lambda x: [i.replace(" ", "") for i in x])

# Assuming you have a DataFrame named 'movies_df'

# Fixing the code for the 'genres' column again (if you want to remove spaces)
movies_df['crew'] = movies_df['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
# Assuming you have a DataFrame named movies_df
movies=movies_df.copy()  # Create a copy with the new name

# Now, you can use the 'movies' DataFrame instead of 'movies_df' for further operations
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: ' '.join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the CountVectorizer with the specified parameters
cv = CountVectorizer(max_features=5000, stop_words='english')

# Fit and transform the 'tags' column of the DataFrame
vectors = cv.fit_transform(new_df['tags']).toarray()

# Check the shape of the resulting array

# Access the vectors for a specific row (e.g., row 9)
row_9_vectors = vectors[9]

# Print the vectors for row 9
import nltk
from nltk.stem.porter import PorterStemmer

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Define a function to perform stemming
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return ' '.join(y)
new_df['tags'] = new_df['tags'].apply(stem)
from sklearn.metrics.pairwise import cosine_similarity


similarity = cosine_similarity(vectors)


def recommend(movie_title):
    # Find the index of the movie with the given title
    movie_index = new_df[new_df['title'] == movie_title].index[0]
    a=[]
    # Get the similarity scores for the specified movie
    distances = similarity[movie_index]

    # Sort the distances in descending order and get the top 4 most similar movies (excluding the movie itself)
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:5]
    for i in movies_list:
        recommended_movie_title = new_df.iloc[i[0]].title
        a.append(recommended_movie_title)

    return a
    # Print the titles of the recommended movies

import streamlit as st
import plotly.express as px

# Define the recommend function here

# Create a Streamlit app
st.title("Stream Cine")

# Create an input field for the user to enter a movie name
user_input = st.text_input("Enter a movie name:")

# Define a button to trigger the recommendation process
if st.button("Get Recommendations"):
    # Call the recommend function with the entered movie name
    recommendations = recommend(user_input)

    # Display the recommendations to the user
    if not recommendations:
        st.write("No recommendations found for this movie.")
    else:
        # Display the recommendations to the user
        st.write("Recommended Movies:")
        for movie in recommendations:
            st.write(movie)
        user_selected_revenue = movies_analysis[movies_analysis['title'] == user_input]['revenue'].values[0]
        other_movies_revenue=movies_analysis[movies_analysis['title'].isin(recommendations)]
        fig = px.bar(
            data_frame=other_movies_revenue,
            x=[user_input] + recommendations,
            y=[user_selected_revenue] + other_movies_revenue['revenue'].tolist(),
            color=[user_input] + recommendations,
            labels={'x': 'Movie Titles', 'y': 'Revenue'},
            title=f'Revenue Comparison: {user_input} vs. Recommended Movies'
        )
        st.plotly_chart(fig)
        import plotly.express as px
        selected_movies_data = movies_analysis[movies_analysis['title'].isin([user_input] + recommendations)]
        fig = px.line(
            data_frame=selected_movies_data,
            x='title',  # x-axis: movie titles
            y=['vote_average', 'vote_count'],  # y-axes: vote_average and vote_count
            labels={'value': 'Value', 'title': 'Movie Titles'},  # Set labels for clarity
            title=f'Vote Comparison: {user_input} vs. Recommended Movies'  # Set a title for the graph
        )
        st.plotly_chart(fig)
        import networkx as nx
        import matplotlib.pyplot as plt
        from sklearn.metrics.pairwise import cosine_similarity

        # Get the 'tags' of the user-selected movie
        selected_movie_tags = movies_analysis[movies_analysis['title'] == user_input]['tags'].values[0]

        # Initialize the CountVectorizer with the same parameters used for other movies
        vectorizer = CountVectorizer(max_features=5000, stop_words='english')

        # Fit the vectorizer on the 'tags' of the other movies
        vectorizer.fit(movies_analysis['tags'])

        # Transform the 'tags' of the user-selected movie using the same vectorizer
        selected_movie_vector = vectorizer.transform([selected_movie_tags]).toarray()

        # Transform the 'tags' of the movies in 's' using the same vectorizer
        similar_movies_tags = movies_analysis[movies_analysis['title'].isin(recommendations)]['tags']
        similar_movies_vectors = vectorizer.transform(similar_movies_tags).toarray()

        # Compute cosine similarity between the user-selected movie and movies in 's'
        cosine_similarities = cosine_similarity(selected_movie_vector, similar_movies_vectors)
        ratings = [1 + (score * 1.5) for score in cosine_similarities[0]]

        # Create a DataFrame with movie titles, ratings, and reverse ordering for the x-axis
        similarity_df = pd.DataFrame({
            'Movie': recommendations,
            'Rating': ratings
        })

        # Sort the DataFrame in reverse order
        similarity_df = similarity_df.sort_values(by='Rating', ascending=False)

        # Add the user input to the DataFrame
        similarity_df = similarity_df.append({'Movie': user_input, 'Rating': 4.0}, ignore_index=True)

        # Create a bar graph using Plotly
        fig = px.bar(
            data_frame=similarity_df,
            x='Movie',
            y='Rating',
            labels={'Movie': 'Movie Titles', 'Rating': 'Rating'},
            title=f'Rating Comparison: {user_input} vs. Recommended Movies',
            color='Movie'
        )

        # Set the x-axis to be in reverse order
        fig.update_xaxes(type='category', categoryorder='total descending')

        # Show the graph
        st.plotly_chart(fig)
