# NLP Project: Web Scraping, Translation, and Summarization

This project showcases Natural Language Processing (NLP) techniques for web scraping, translation, and extractive summarization. The code includes examples and models for each of these tasks.

## Web Scraping

### Overview
Web scraping is the process of extracting information from websites. In this project, we use the `requests` library to fetch HTML content and `BeautifulSoup` for parsing and navigating the HTML.

### Usage
1. Install the required dependencies:
   ```bash
   pip install requests beautifulsoup4
   ```
2. Run the web scraping script:
   ```bash
   python web_scraping.py
   ```

## Translation

### Overview
Translation is performed using the `googletrans` library. It allows us to translate text from one language to another. The `Translator` class is used for this purpose.

### Usage
1. Install the required dependency:
   ```bash
   pip install googletrans==4.0.0-rc1
   ```
2. Run the translation script:
   ```bash
   python translation.py
   ```

## Extractive Summarization

### Overview
Extractive summarization involves identifying the most important sentences from a document. In this project, we use spaCy and NLTK to tokenize, filter, and rank sentences based on importance.

### Usage
1. Install the required dependencies:
   ```bash
   pip install spacy nltk
   python -m spacy download en_core_web_sm
   python -m nltk.downloader punkt
   ```
2. Run the extractive summarization script:
   ```bash
   python extractive_summarization.py
   ```

## Model Training (Seq2Seq Encoder-Decoder)

### Overview
A Seq2Seq model with an Encoder-Decoder architecture is trained for sequence-to-sequence tasks. In this project, the model is trained for language translation.

### Dependencies
- tensorflow
- keras
- nltk

### Usage
1. Install the required dependencies:
   ```bash
   pip install tensorflow keras nltk
   python -m nltk.downloader punkt
   ```
2. Run the model training script:
   ```bash
   python model_training.py
   ```
