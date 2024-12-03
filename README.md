# Music Recommendation System

A Streamlit-based web application that provides personalized music recommendations based on user preferences and song features. The system analyzes Spotify streaming data and uses content-based filtering to suggest similar songs.

## Features

- **Song Search**: Search for songs or artists in the database
- **Personalized Recommendations**: Get song recommendations based on your selected track
- **Data Visualizations**:
  - Top 10 Most Streamed Songs on Spotify
  - Feature Correlation Heatmap
  - Feature Distribution Analysis
  - Artist Analysis with Top Artists
- **Album Cover Integration**: Display album artwork for recommended songs using Spotify API
- **User Feedback System**: Collect and store user feedback for system improvement

## Prerequisites

Before running this application, make sure you have Python 3.7+ installed and the following dependencies:

```bash
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
spotipy
requests
Pillow
```

## Installation

1. Clone this repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up Spotify API credentials:
   - Visit [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
   - Create a new application
   - Get your Client ID and Client Secret
   - Add them to the designated spots in `app.py`:
     ```python
     SPOTIPY_CLIENT_ID = 'your_client_id_here'
     SPOTIPY_CLIENT_SECRET = 'your_client_secret_here'
     ```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Use the application:
   - Search for a song or artist using the search bar
   - Select a song from the dropdown menu
   - Click "Get Recommendations" to see similar songs
   - Explore various data visualizations
   - Provide feedback using the feedback form

## Data Structure

The application expects a CSV file with the following columns:
- Track
- Album Name
- Artist
- Release Date
- ISRC
- All Time Rank
- Track Score
- Spotify Streams
- Spotify Playlist Count
- Spotify Playlist Reach
- Spotify Popularity
- YouTube Views
- YouTube Likes
- TikTok Posts
- TikTok Likes
- TikTok Views
- Additional metrics...

## How It Works

1. **Data Loading**: 
   - Loads music data from CSV file
   - Handles different file encodings
   - Preprocesses data and handles missing values

2. **Recommendation System**:
   - Uses content-based filtering
   - Implements cosine similarity for finding similar songs
   - Considers features like Track Score, Spotify Popularity, YouTube Views, and TikTok Views

3. **Visualization**:
   - Creates interactive visualizations using matplotlib and seaborn
   - Displays dataset statistics and trends
   - Shows artist analysis and feature correlations

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Error Handling

The application includes robust error handling for:
- File loading issues
- Data preprocessing errors
- API connection problems
- Missing values in features
- Image fetching errors

## Future Improvements

- Implement collaborative filtering
- Add more advanced visualization options
- Integrate more music streaming platforms
- Enhance the recommendation algorithm
- Add user authentication
- Implement a caching system for album covers
- Add playlist generation feature

## Course Information

This project is part of the BICT242: Data Scalability and Analytics course.

## License

None

## Contact

Email: mvuleniwilliam@gmail.com
Alternative email: williammnisi@outlook.com

## Acknowledgments

- Spotify Web API
- Streamlit Documentation
- scikit-learn Documentation
- Dr. Norwell Zhakata
