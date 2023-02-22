from flask import Flask, request, render_template
from flask_ngrok import run_with_ngrok
from langchain.chains import llm
from ast import Index
from gpt_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
from waitress import serve
import sys
import os

app = Flask(__name__)
run_with_ngrok(app)

os.environ["OPENAI_API_KEY"] = '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'

import json

# Load the index data from the file
index_filename = 'index.json'
with open(index_filename, 'r') as f:
    index_data = json.load(f)

# Create the GPT index object
index = GPTSimpleVectorIndex(index_data)

# Initialize the history list
history = []

@app.route('/', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        query = request.form['query']
        best_answer = index.query(query, response_mode="compact", verbose=True)
        
        # Add the current query and response to the history list
        history.append((query, best_answer))
        
        # Reverse the order of the history list so that the most recent queries are at the top
        history.reverse()
        
        return render_template('index.html', answer=best_answer, history=history)
    
    # Reverse the order of the history list so that the most recent queries are at the top
    history.reverse()
    
    return render_template('index.html', history=history)

if __name__ == '__main__':
    #app.run()
    serve(app)
