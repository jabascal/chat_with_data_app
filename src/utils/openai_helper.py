import json
import os

import openai


def get_completion(prompt, model='gpt-3.5-turbo'):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message["content"]


# Retrieve and set API KEY
def read_key_from_file(path_file, name_file_key):
    with open(os.path.join(path_file, name_file_key), 'r') as f:
        org_data = json.load(f)
        
    openai.organization = org_data['organization']
    openai.api_key = org_data['api_key']