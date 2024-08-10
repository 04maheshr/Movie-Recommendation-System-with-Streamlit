# Stream Cine

## Overview

**Stream Cine** is a movie recommendation and analysis application built using Python, Streamlit, and various data processing libraries like Pandas, NumPy, and Scikit-learn. The app provides movie recommendations based on cosine similarity of movie tags and displays analysis on revenue and vote comparisons between user-selected movies and the recommended movies. The project leverages machine learning and natural language processing to analyze movie data and generate meaningful insights.

## Features

- **Movie Recommendations:** Get movie recommendations based on the cosine similarity of movie tags.
- **Revenue Comparison:** Visualize the revenue comparison between the selected movie and the recommended movies.
- **Vote Analysis:** Analyze and compare the vote average and vote count of the selected movie and its recommendations.
- **Rating Comparison:** View a custom rating comparison between the selected movie and the recommended movies.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/stream-cine.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd stream-cine
    ```

3. **Install the required dependencies:**

    Make sure you have Python installed. Then, install the required packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the application:**

    Use the following command to run the Streamlit application:

    ```bash
    streamlit run ap.py
    ```

2. **Interacting with the App:**

    - Enter the name of a movie in the text input field.
    - Click on the "Get Recommendations" button to get movie recommendations.
    - View the recommended movies, revenue comparison, vote analysis, and rating comparison in the generated visualizations.

## Project Structure

- **ap.py:** The main script containing the logic for data processing, movie recommendation, and visualization.
- **credits.csv:** A dataset containing movie credits (cast and crew information).
- **movies.csv:** A dataset containing movie details (title, overview, genres, etc.).

## Key Components

- **Pandas:** For data manipulation and analysis.
- **Scikit-learn:** For implementing the cosine similarity algorithm.
- **Streamlit:** For creating an interactive web application.
- **Plotly:** For creating interactive visualizations (bar graphs, line graphs).
- **NLTK:** For text processing (stemming).

## Data Sources

The data used in this project includes movie details and credits, typically sourced from publicly available datasets such as those from [The Movie Database (TMDb)](https://www.themoviedb.org/).

## Future Enhancements

- **Enhanced Recommendation Algorithms:** Experiment with different algorithms for recommendations, such as collaborative filtering.
- **User Ratings:** Allow users to rate movies and refine recommendations based on user preferences.
- **Expanded Visualizations:** Include additional metrics and visualizations for deeper analysis.

## Contributing

Feel free to fork this repository, create a new branch, and submit a pull request with your contributions. All contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Streamlit:** For providing an easy-to-use framework for building web applications in Python.
- **TMDb:** For the comprehensive movie datasets used in this project.

