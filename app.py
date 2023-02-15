from flask import Flask, request, render_template
from flask_ngrok import run_with_ngrok
from langchain.chains import llm
from ast import Index
from gpt_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import sys
import os

app = Flask(__name__)
run_with_ngrok(app)

os.environ["OPENAI_API_KEY"] = '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'

import json

def load_index():
    with open('index.json', 'r') as f:
        index_dict = json.load(f)
    index = GPTListIndex(index_dict)
    return index

index = load_index()

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
    app.run()
