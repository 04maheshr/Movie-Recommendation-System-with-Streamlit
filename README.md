# Stream Cine - Movie Recommendation System

## Overview
Stream Cine is a movie recommendation system that uses cosine similarity to recommend movies based on various features such as keywords, cast, crew, genres, and overview. It provides visualizations to compare the user-selected movie with the recommended movies in terms of revenue, vote average, vote count, and similarity ratings.

## Features
- **Movie Recommendations**: Get movie recommendations based on a user-selected movie.
- **Revenue Comparison**: Visualize the revenue comparison between the user-selected movie and recommended movies.
- **Vote Comparison**: Compare the vote average and vote count between the user-selected movie and recommended movies.
- **Similarity Ratings**: Visualize similarity ratings between the user-selected movie and recommended movies.

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/username/repository.git
    cd repository
    ```

2. Install the required libraries:
    ```sh
    pip install pandas numpy scikit-learn nltk streamlit plotly networkx
    ```

3. Download and place the CSV files (`credits.csv` and `movies.csv`) in the project directory.

## Usage
1. Run the Streamlit app:
    ```sh
    streamlit run ap.py
    ```

2. Open your browser and navigate to the provided local URL (e.g., `http://localhost:8501`).

3. Enter a movie name in the input field and click "Get Recommendations" to see the recommended movies and visualizations.

## File Structure
- `ap.py`: Main script containing the logic for data processing, movie recommendation, and Streamlit app.
- `credits.csv`: CSV file containing movie credits data.
- `movies.csv`: CSV file containing movies data.

## Code Explanation
### Data Loading
Reads the CSV files `credits.csv` and `movies.csv` into DataFrames `credits_df` and `movies_df`.

### Data Merging and Cleaning
Merges the DataFrames on the `title` column and selects relevant columns. Drops missing values and cleans up the data.

### Data Transformation
- Extracts names from JSON-like strings in `genres` and `keywords`.
- Limits the number of names in `cast` to the first three.
- Extracts the director's name from `crew`.
- Creates a `tags` column by concatenating relevant columns and converting text to lowercase.

### Vectorization
Uses `CountVectorizer` to convert the `tags` text into numerical data for similarity calculations.

### Stemming
Applies stemming to the `tags` column to reduce words to their root forms.

### Similarity Calculation
Calculates cosine similarity between the vectorized tags of all movies.

### Recommendation Function
The `recommend` function takes a movie title as input, finds the most similar movies based on cosine similarity, and returns a list of recommended movie titles.

### Streamlit App
Provides an interactive interface for users to enter a movie name and get recommendations. Displays visualizations comparing the user-selected movie with recommended movies.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Acknowledgements
- The data used in this project is sourced from [TMDb](https://www.themoviedb.org/).

## Contact
For any inquiries or issues, please contact [your-email@example.com](mailto:your-email@example.com).
