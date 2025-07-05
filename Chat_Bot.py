import os
import re
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import gradio as gr


# Load the sentence transformer model
# Determine the device to use
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

class ICSKnowledgeBase:
    def __init__(self):
        self.sections = {}
        self.qa_pairs = []
        self.embeddings = None
        self.texts = []
        
    def load_data(self, scraped_file: str, qa_file: str):
        """Load data from both files"""
        self._load_scraped_data(scraped_file)
        self._load_qa_data(qa_file)
        self._create_embeddings()
    
    def _load_scraped_data(self, file_path: str):
        """Process the scraped data file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content into sections based on headings
        sections = re.split(r'\n([A-Z][A-Z\s]+)\n', content)
        
        # The first item is usually not a section
        for i in range(1, len(sections), 2):
            section_name = sections[i].strip()
            section_content = sections[i+1].strip()
            self.sections[section_name] = section_content
    
    def _load_qa_data(self, file_path: str):
        """Process the Q&A data file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content into Q&A pairs
        qa_pairs = re.split(r'\n\n', content.strip())
        
        for pair in qa_pairs:
            # Split into question and answer
            parts = re.split(r'\n', pair, maxsplit=1)
            if len(parts) == 2:
                question = parts[0].strip()
                answer = parts[1].strip()
                self.qa_pairs.append((question, answer))
    
    def _create_embeddings(self):
        """Create embeddings for all text chunks"""
        self.texts = []
        
        # Add Q&A pairs
        for question, answer in self.qa_pairs:
            self.texts.append(f"Q: {question}\nA: {answer}")
        
        # Add section content
        for section, content in self.sections.items():
            # Split content into smaller chunks if too long
            chunks = self._split_text(content)
            for chunk in chunks:
                self.texts.append(f"Section: {section}\nContent: {chunk}")
        
        # Create embeddings
        self.embeddings = model.encode(self.texts, convert_to_tensor=True, device=device)
    
    def _split_text(self, text: str, max_length: int = 500) -> List[str]:
        """Split long text into chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(word)
            current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search for relevant information"""
        query_embedding = model.encode(query, convert_to_tensor=True, device=device)
        
        # Compute similarity
        similarities = cosine_similarity(
            query_embedding.unsqueeze(0).cpu(),
            self.embeddings.cpu()
        )[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [(self.texts[i], similarities[i]) for i in top_indices]
        
        return results
    
    def get_answer(self, question: str) -> str:
        """Get answer for a question"""
        # First check if it's a direct Q&A match
        for q, a in self.qa_pairs:
            if q.lower() == question.lower():
                return a
        
        # Otherwise do semantic search
        results = self.search(question)
        
        if not results:
            return "I couldn't find information about that. Please try rephrasing your question."
        
        # Combine top results
        answer = "Here's what I found:\n\n"
        for i, (text, score) in enumerate(results, 1):
            answer += f"{i}. {text}\n\n"
        
        return answer

# Initialize the knowledge base
knowledge_base = ICSKnowledgeBase()

# Load the data files
scraped_file = "C:\\Users\\anuj2\\Downloads\\ics_data_final.txt"
qa_file = "C:\\Users\\anuj2\\Downloads\\Questions and Answers ics.txt"

if os.path.exists(scraped_file) and os.path.exists(qa_file):
    knowledge_base.load_data(scraped_file, qa_file)
else:
    raise FileNotFoundError("Required data files not found. Please ensure both files are in the correct location.")

def respond(message, history):
    """Function to handle chatbot responses"""
    response = knowledge_base.get_answer(message)
    return response

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# ICS IIT Kanpur Chatbot")
    gr.Markdown("Ask me anything about the Institute Counselling Service at IIT Kanpur.")
    
    chatbot = gr.Chatbot(height=300)
    msg = gr.Textbox(label="Your Question")
    clear = gr.Button("Clear")
    
    def user(user_message, chat_history):
        return "", chat_history + [[user_message, None]]
    
    def bot(chat_history):
        bot_message = respond(chat_history[-1][0], chat_history)
        chat_history[-1][1] = bot_message
        return chat_history
    
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(share = True)