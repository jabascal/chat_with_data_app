import gradio as gr
from utils.langchain_helper import init_embedding, read_split_doc, create_db_from_documents, init_llm_qa_chain

    
# Interface
def chat_to_your_data_ui(openai_api_key, doc_type, doc_path, chunk_size, chunk_overlap,
                         llm_name, temperature, share_gradio, image_path):

    # Define embedding
    embedding = init_embedding(openai_api_key=openai_api_key)    

    with gr.Blocks(theme=gr.themes.Glass()) as demo:
        # Description            
        gr.Markdown(
        """
        # Chat to your data
        Ask questions to the chatbot about your document. The chatbot will find the answer to your question. 
        You can modify the document type and provide its path/link. A document has been preselected. 
        You may also modify some of the advanced options.        
        """)

        with gr.Row():
            with gr.Column():
                # -------------------------
                # Parameters
                # Temperature and document type
                gr.Markdown(
                """
                ## Select parameters
                Default parameters are already provided.
                """
                )
                # Advanced parameters (hidden)
                with gr.Accordion(label="Advanced options",open=False):
                    gr.Markdown(
                    """
                    The document is split into chunks, keeping semantically related pieces together and with some overlap. 
                    You can modify the chunk size and overlap. The temperature is used to control the randomness of the output 
                    (the lower the temperature the more deterministic the ouput, the higher its value the more random the result, with $temperature\in[0,1]$).
                    """
                    )        
                    sl_temperature = gr.Slider(minimum=0.0, maximum=1.0, value=temperature, label="Temperature", 
                                                scale=2)
                    with gr.Row():
                        num_chunk_size = gr.Number(value=chunk_size, label="Chunk size", scale=1)
                        num_chunk_overlap = gr.Number(value=chunk_overlap, label="Chunk overlap", scale=1)


            # Chatbot image
            # https://drive.google.com/file/d/1HDnBsdfUYrCHOFtP2-DqomcmBSs9XyNI/view?usp=sharing
            # ![](https://drive.google.com/uc?id=1HDnBsdfUYrCHOFtP2-DqomcmBSs9XyNI)
            gr.Markdown(
            f"""
            <img src="{image_path}" alt="drawing" width="300"/>            
            """, scale=1)

        # -------------------------
        # Select document
        gr.Markdown(
        """
        ## Select document
        Select the document type and provide its path/link. A document has been preselected.
        """)
        with gr.Row():
            drop_type = gr.Dropdown(["url", "pdf", "youtube"], 
                                    label="Document Type", value=doc_type, min_width=30, scale=1)
            text_path = gr.Textbox(label="Document Path/URL", placeholder=doc_path, scale=5)
        
        with gr.Row():
            # Read document
            btn_read = gr.Button("Read document")
            text_read_output = gr.Textbox(label="Reading state", interactive=False, placeholder="Finished reading document! Let's chat!")
        # -------------------------
        # Chatbot
        gr.Markdown("""
        ## Chatbot  
        To chat, introduce a question and press enter.
                    
        Question examples:
                    
         - Hi
                    
         - What is the document about?
                    
         - What can visit in Lyon?                   
        """
        )
        # Chatbot
        chatbot = gr.Chatbot()
        
        # Input message
        msg = gr.Textbox(label="Question")
        
        # Clear button
        clear = gr.Button("Clear")

        # -------------------------
        # Init the LLM and read document
        def init_read_doc(doc_type, doc_path, chunk_size, chunk_overlap, temperature):
            global qa_chain
            # Read and split document using langchain
            print(f"Reading document {doc_path} of type {doc_type} ...")
            docs_split = read_split_doc(doc_type, doc_path, chunk_size, chunk_overlap)
            # -------------------------
            # Create vector database from data    
            db = create_db_from_documents(docs_split, embedding)
            # -------------------------
            # Init the LLM and qa chain
            llm, qa_chain, memory = init_llm_qa_chain(llm_name, temperature, openai_api_key, db)            
        
        # Init the LLM and read document with default parameters
        init_read_doc(doc_type, doc_path, chunk_size, chunk_overlap, temperature)
        # -------------------------
        # When read document (aready read with default parameters)
        def reading_doc_msg(doc_type, doc_path):
            return f"Reading document {doc_path} of type {doc_type} ..."
        def read_doc_msg():
            return "Finished reading the document! Let's chat!"
        def clear_chatbot():            
            return "", ""

        btn_read.click(reading_doc_msg,                                         # Reading message 
                            inputs=[drop_type, text_path], 
                            outputs=text_read_output).then(init_read_doc,   # Init qa chain and read document
                                inputs=[drop_type, text_path, 
                                        num_chunk_size, num_chunk_overlap,
                                        sl_temperature], 
                                queue=False).then(read_doc_msg,             # Finished reading message
                                        outputs=text_read_output).then(clear_chatbot, # Clear chatbot
                                                outputs=[chatbot, msg], queue=False)        
        # -------------------------
        # When question 
        def qa_input_msg_history(question, chat_history):
            # QA function that inputs the answer and the history.
            # History managed internally by ChatInterface            
            answer = qa_chain({"question": question})['answer']
            #response = qa_chain({"question": input})
            chat_history.append((question, answer))
            return "", chat_history

        msg.submit(qa_input_msg_history, 
                     inputs=[msg, chatbot], 
                     outputs=[msg, chatbot], queue=False)#.then(bot, chatbot, chatbot)
        
        # When clear
        clear.click(lambda: None, None, chatbot, queue=False)
    #demo.queue() # To use generator, required for streaming intermediate outputs
    demo.launch(share=share_gradio)


# Simple Gradio chat interface
def chat_interface(chat_examples, chat_description, qa_chain, share_gradio):
    def qa_input_msg_history(input, history, temperature):
        # QA function that inputs the answer and the history.
        # History managed internally by ChatInterface
        #answer = qa_answer(input)
        answer = qa_chain({"question": input})['answer']
        return answer

    demo = gr.ChatInterface(fn=qa_input_msg_history, 
                    title="Chat to your data",
                    description=chat_description,
                    examples=chat_examples,
                    retry_btn=None,
                    undo_btn=None,
                    ).launch(share=share_gradio)