# Databricks notebook source
iface = gr.Interface(
    fn=rag_chat,
    inputs=gr.Textbox(label='Question'),
    outputs=gr.Textbox(label='Response'),
    title="Databricks Serving Endpoint Demo",
    description="Enter your question and click Submit to make a prediction",
    theme="freddyaboulton/dracula_revamped" #huggingface template
)

# Launch the Gradio interface
iface.launch(share=True, debug=True) # for some reason need to run in debug mode to work
