# Pulpnet : A project on Neural Networks

## ICS IIT Kanpur Chatbot

### Overview  
This project consists of a chatbot designed to answer questions about the Institute Counselling Service (ICS) at IIT Kanpur. The system combines web scraping and natural language processing to provide accurate responses to user queries.

---

### Features :
- Web scraping of ICS website content (20+ pages)  
- PDF brochure text extraction  
- Question-answering based on semantic search  
- Gradio-based web interface for easy interaction  
- GPU acceleration support (if available)  

---

### Files:

**`Scraping_data.py`**
- Handles web scraping and data collection  
- Scrapes text content from multiple ICS web pages  
- Extracts text from the ICS brochure PDF  
- Saves combined data to `"ics_data_final.txt"`  

**`Chat_Bot.py`**
- Implements the chatbot functionality  
- Loads scraped data and Q&A pairs  
- Uses `sentence-transformers` for semantic search  
- Provides a Gradio web interface for interaction  

---

### Requirements

- Python 3.7+

**Required packages:**
- `requests`  
- `beautifulsoup4`  
- `pypdf`  
- `numpy`  
- `torch`  
- `sentence-transformers`  
- `scikit-learn`  
- `gradio`  

---

### Data Files

- `ics_data_final.txt`: Combined scraped data from ICS website  
- `Questions and Answers ics.txt`: Predefined Q&A pairs (includes general questions to improve chatbot accuracy)  

---

### Usage

```bash
# Step 1: Run the scraper to collect website and PDF data
python Scraping_data.py

# Step 2: Launch the chatbot interface
python Chat_Bot.py

# The Gradio interface will open in your browser or provide a shareable link



