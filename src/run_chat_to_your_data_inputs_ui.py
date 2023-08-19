# Description: Run QA to your data
#
# Read and process data:
#  - Read data from file or db (already processed)
#  - Split into chunks
#  - Create vector database from data
# 
# Find relevant documents
#  - Prompt the user to introduce a question
#  - Find relevant documents
#  - Reduce documents
#
# QA CHAIN
#  - Prompt the user to introduce a question
#  - Run QA chain


import os

import openai

from utils.in_out_helper import load_config
from utils.app import chat_to_your_data_ui
from utils.openai_helper import read_key_from_file

# Working directory
print(f"Working directory: {os.getcwd()}")

# Import YAML parameters from config/config.yaml
config_file = "config/config.yaml"

# Load config file 
param = load_config(config_file)

def main(param):
    # PARAMETERS
    #
    # OpenAI API KEY
    path_file_key = param['openai']["path_file_key"]
    name_file_key = param['openai']["name_file_key"]
    # LLM
    temperature = param['llm']["temperature"]
    llm_name = param['llm']["llm_name"]
    # Document
    doc_type = param['doc']["doc_type"]
    doc_path = param['doc']["doc_path"]
    chunk_overlap = param['doc']["chunk_overlap"]
    chunk_size = param['doc']["chunk_size"]
    # Database
    persist_path = param['db']["persist_path"]
    mode_input = param['db']["mode_input"]
    # Chat
    chat_examples = param['chat']["chat_examples"]
    chat_description = param['chat']["chat_description"]
    share_gradio = param['chat']["share_gradio"]
    # Chatbot image
    chatbot_image = param['image']["image_path"]
    # ---------------------------------------------------------
    # Read OpenAI key from filepath_file
    read_key_from_file(path_file_key, name_file_key)
    # ---------------------------------------------------------
    # Interface to chat with your data
    chat_to_your_data_ui(openai.api_key, doc_type, doc_path, chunk_size, chunk_overlap,
                         llm_name, temperature, share_gradio, chatbot_image)

if __name__ == "__main__":
    main(param)