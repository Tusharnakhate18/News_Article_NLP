# News Article Summary App
## Overview
The News Article Summary App is a Streamlit web application designed to fetch news articles from a given URL, summarize the content of the articles using BART (Bidirectional and Auto-Regressive Transformers) model, and perform analysis on the summarized text. The application also provides text classification based on trending keywords to identify whether an article is considered "hot" or not.

## Features
- **Fetch News:** Users can input a URL to fetch news articles from various sources.
- **Text Summarization:** The application summarizes the fetched news articles using the BART model, providing concise summaries for quick understanding.
- **Text Classification:** Performs sentiment analysis on the summarized text to determine the sentiment score and categorizes articles as "hot" or "not hot" based on predefined thresholds.
## Setup and Installation
- 1. Clone the repository:
( git clone https://github.com/yourusername/News-Summary-App.git)
- 2. Install the required dependencies:
( pip install -r requirements.txt)
- 3. Run the Streamlit application:
( streamlit run app.py)
- 4. Access the application through the provided local server URL.
## Usage
- Enter the URL of the news website or article in the text input box.
- Click on the "Fetch News" button to retrieve the news articles from the provided URL.
- Use the "Text Summarization" button to generate summaries of the fetched news articles.
- Utilize the "Text Classification" button to perform sentiment analysis and classify articles as "hot" or "not hot".
## Dependencies
- Streamlit
- NLTK
- Transformers (Hugging Face)
- Summa
- BeautifulSoup
- Requests
- Pytz
- Gensim
- Pandas
- NumPy
- Authors
- Your Name
## License
This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgments
The developers of Streamlit, NLTK, Transformers, Summa, BeautifulSoup, and other libraries used in this project.
Special thanks to Hugging Face for providing pre-trained language models and tokenizers.







