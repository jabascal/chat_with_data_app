---
title: Chat With Data App
emoji: 📉
colorFrom: red
colorTo: pink
sdk: gradio
sdk_version: 3.40.1
app_file: app.py
pinned: false
license: mit
---

# Chat with data app    
[![Open in Spaces](https://badgen.net/static/open/on%20HFSpaces/cyan)](https://replit.com/@jabascal1/ytube-download?v=1) [![Open tutorial in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jabascal/chat_with_data_app/blob/main/notebook/chat_to_your_data_medium.ipynb)


Chatbot app to chat with any source of data (doc, URL, audio, ...) leveraging LLMs and LangChain. Current version has the following feautures:
- LLMs: OpenAI GPT-3.5. It requires providing openai APY_KEY
- Data source: pdf, URL, youtube

![](https://github.com/jabascal/chat_with_data_app/blob/main/figures/app_ui.png)

This app has been inspired on DeepLearning AI course [LangChain:Chat with your data](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data)

## Installation and execution
Install in a python environment:
```
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cd src/
    python run_chat_to_your_data_inputs_ui.py
```
