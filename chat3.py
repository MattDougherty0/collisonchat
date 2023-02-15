!pip install Flask
!pip install gpt_index
!pip install ngrok
!pip install langchain
!pip install openAI


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

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 256
    max_chunk_overlap = 20
    chunk_size_limit = 600

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(
      documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper, verbose=True
      
  )
    
    index.save_to_disk('index.json')

    return index

book_path = input("Enter the path to the directory containing your documents: ")
index = construct_index(book_path)

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
