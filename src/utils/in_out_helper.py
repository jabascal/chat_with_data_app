import re

import yaml

def pretty_print_docs(docs, question = None):
    print(f"\n{'-' * 100}\n")
    if question:
        print(f"Question: {question}")

    print("".join([f"Document {i+1}:\n\nMetadata: {d.metadata}\n" + d.page_content for i, d in enumerate(docs)]))


# Clear white lines in web pages
def clear_blank_lines(docs):
    for doc in docs:
        doc.page_content = re.sub(r"\n\n\n+", "\n\n", doc.page_content)
    return docs
    
def load_config(config_file):
    # Import YAML parameters from config/config.yaml

    # define custom tag handler
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return ''.join([str(i) for i in seq])
    # add string concatenation function when parse the yaml file
    def concat(loader, node):
        seq = loader.construct_sequence(node)
        seq = [str(tmp) for tmp in seq]
        return ''.join(seq)

    def get_loader():
        # Add constructors to the loader
        loader = yaml.SafeLoader
        loader.add_constructor('!join', join)
        loader.add_constructor('!concat', concat)
        return loader

    with open(config_file, 'r') as stream:        
        param = yaml.load(stream, Loader=get_loader())
        print(yaml.dump(param))
    return param
