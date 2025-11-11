## Movie Recommendation System

This project is a movie recommendation system built in Python. It uses the **Flask framework** to provide a simple web interface for generating recommendations.

All recommendation algorithms and similarity metrics are implemented manually, without relying on external machine learning libraries or databases.

### Features

The system implements several recommendation algorithms and similarity metrics.

**Algorithms Implemented:**

  * **User-User Collaborative Filtering** (`algorithm=user`)
  * **Item-Item Collaborative Filtering** (`algorithm=item`)
  * **Tag-Based Recommendation** (`algorithm=tag`)
  * **Hybrid Recommendation:** A custom combination of the three algorithms above.

**Similarity Metrics Supported:**

  * `jaccard`
  * `dice`
  * `cosine`
  * `pearson`

-----

### Setup and Installation

1.  **Python:** The project was developed using **Python 3.11.3**.
2.  **Dependencies:** Install the required Flask framework.
    ```bash
    pip install Flask
    ```
3.  **Dataset:**
      * Download the **ml-latest.zip** dataset from the Movielens website.
      * Fix the path for the data to be found by the application.

-----

### How to Run

1.  Navigate to the project's root directory in your terminal.
2.  Run the main application:
    ```bash
    python Main.py
    ```
3.  Once preprocessing is complete, the Flask server will start and display the local address it is running on. You will see a message similar to this in your terminal:
    ```
    * Running on http://127.0.0.1:5000
    ```
4.  Open the specific `http://...` address shown in your terminal output in your web browser to use the system.

-----

### Using the Web Interface

1.  Use the dropdown menus to select the desired **Algorithm** and **Similarity Metric**.
2.  Enter the appropriate ID in the input field and click "Get Recommendations."

> **Input Guide:**
>
>   * For **User-User**, **Item-Item**, and **Hybrid** algorithms, the input is a **User ID**.
>   * For the **Tag** algorithm, the input must be a **Movie ID**.
>
> **Performance Note:**
>
>   * Please allow **30-60 seconds** for the **Hybrid** algorithm to process and return recommendations.
