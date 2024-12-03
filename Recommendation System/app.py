import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
import csv


@st.cache_data
def load_data():
    # Define column names based on the CSV structure
    columns = [
        'Track', 'Album Name', 'Artist', 'Release Date', 'ISRC', 'All Time Rank', 'Track Score',
        'Spotify Streams', 'Spotify Playlist Count', 'Spotify Playlist Reach', 'Spotify Popularity',
        'YouTube Views', 'YouTube Likes', 'TikTok Posts', 'TikTok Likes', 'TikTok Views',
        'YouTube Playlist Reach', 'Apple Music Playlist Count', 'AirPlay Spins', 'SiriusXM Spins',
        'Deezer Playlist Count', 'Deezer Playlist Reach', 'Amazon Playlist Count', 'Pandora Streams',
        'Pandora Track Stations', 'Soundcloud Streams', 'Shazam Counts', 'TIDAL Popularity', 'Explicit Track'
    ]

    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

    for encoding in encodings:
        try:
            df = pd.read_csv("Most Streamed Spotify Songs 2024.csv",
                             names=columns,
                             quoting=csv.QUOTE_MINIMAL,
                             quotechar='"',
                             delimiter=',',
                             encoding=encoding)
            st.success(f"Successfully loaded the CSV file with {encoding} encoding.")
            break
        except UnicodeDecodeError:
            continue
    else:
        st.error("Failed to load the CSV file. Please check the file encoding.")
        return None

    # Convert numeric columns and handle NaN values
    numeric_columns = ['All Time Rank', 'Track Score', 'Spotify Streams', 'Spotify Playlist Count',
                       'Spotify Playlist Reach', 'Spotify Popularity', 'YouTube Views', 'YouTube Likes',
                       'TikTok Posts', 'TikTok Likes', 'TikTok Views']

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

    # Impute NaN values with mean
    imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

    return df


# Load data
df = load_data()

if df is not None:
    # Display dataset info
    st.write("First few rows of the dataset:")
    st.write(df.head())
    st.write("Column names in the dataset:")
    st.write(df.columns.tolist())

    # Identify available features for recommendation
    available_features = ['Track Score', 'Spotify Popularity', 'YouTube Views', 'TikTok Views']
    available_features = [f for f in available_features if f in df.columns]

    if not available_features:
        st.error("No expected features found in the dataset. Please check your CSV file.")
    else:
        st.write("Available features for recommendation:", available_features)

        # Feature scaling with NaN handling
        scaler = MinMaxScaler()
        df[available_features] = scaler.fit_transform(df[available_features])


    # Recommendation function
    def get_recommendations(song_name, df, features, n=5):
        idx = df[df['Track'] == song_name].index[0]
        song_vector = df.loc[idx, features].values.reshape(1, -1)

        # Check for NaN values in the features
        if pd.isna(song_vector).any() or pd.isna(df[features].values).any():
            st.warning("Some features contain NaN values. Recommendations may be less accurate.")

            # Impute NaN values with mean for the song vector and feature matrix
            imputer = SimpleImputer(strategy='mean')
            song_vector = imputer.fit_transform(song_vector)
            feature_matrix = imputer.fit_transform(df[features].values)
        else:
            feature_matrix = df[features].values

        similarity_scores = cosine_similarity(song_vector, feature_matrix)
        similar_indices = similarity_scores.argsort()[0][::-1][1:n + 1]
        return df.iloc[similar_indices][['Track', 'Artist', 'Spotify Streams']]


    # Streamlit app
    st.title('Music Recommendation System')

    # Sidebar
    st.sidebar.title('Dataset Statistics')
    st.sidebar.write(f"Total number of songs: {len(df)}")
    st.sidebar.write(f"Number of unique artists: {df['Artist'].nunique()}")
    st.sidebar.write(f"Average Spotify streams per song: {df['Spotify Streams'].mean():,.0f}")

    # Search functionality
    search_query = st.text_input('Search for a song or artist:')
    if search_query:
        search_results = df[
            df['Track'].str.contains(search_query, case=False) | df['Artist'].str.contains(search_query, case=False)]
        if not search_results.empty:
            st.write(f"Search results for '{search_query}':")
            display_columns = ['Track', 'Artist', 'Spotify Streams']
            st.dataframe(search_results[display_columns])
            selected_song = st.selectbox('Select a song for recommendations:', search_results['Track'].tolist())
        else:
            st.write("No results found. Please try a different search term.")
    else:
        selected_song = st.selectbox('Choose a song:', df['Track'].tolist())

    # Get recommendations
    if st.button('Get Recommendations'):
        if available_features:
            recommendations = get_recommendations(selected_song, df, available_features)
            st.write(f"Recommendations for '{selected_song}':")
            st.table(recommendations)
        else:
            st.error("Cannot generate recommendations due to missing features.")

    # Visualizations
    st.header('Data Visualizations')

    # Top 10 Most Streamed Songs
    st.subheader('Top 10 Most Streamed Songs on Spotify')
    top_10_songs = df.nlargest(10, 'Spotify Streams')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Spotify Streams', y='Track', data=top_10_songs, ax=ax)
    plt.title('Top 10 Most Streamed Songs on Spotify')
    st.pyplot(fig)

    # Feature Correlation Heatmap
    if available_features:
        st.subheader('Feature Correlation Heatmap')
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[available_features].corr(), annot=True, cmap='coolwarm', ax=ax)
        plt.title('Correlation Heatmap of Song Features')
        st.pyplot(fig)

    # Feature Distribution
    if available_features:
        st.subheader('Feature Distribution')
        fig, ax = plt.subplots(figsize=(12, 6))
        df[available_features].boxplot(ax=ax)
        plt.title('Distribution of Song Features')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Artist Analysis
    st.header('Artist Analysis')

    # Top Artists by Number of Songs
    top_artists = df['Artist'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=top_artists.values, y=top_artists.index, ax=ax)
    plt.title('Top 10 Artists by Number of Songs')
    plt.xlabel('Number of Songs')
    st.pyplot(fig)

    # User Feedback
    st.header('Feedback')
    user_feedback = st.text_area("Please provide any feedback or suggestions for improving the recommendation system:")
    if st.button('Submit Feedback'):
        st.write("Thank you for your feedback! We'll use it to improve our system.")

    # Add information about the project
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This Music Recommendation System is a project for the BICT242: Data Scalability and Analytics course. It uses content-based filtering to suggest songs based on audio features and popularity metrics.")

else:
    st.error("Unable to proceed due to data loading error.")