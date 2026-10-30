from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from dotenv import load_dotenv
from screen import screen_resumes_stream

# 1. Custom CSS to add space between columns
custom_css = """
#left-column {
    padding-right: 40px; /* This creates the space between the columns */
}
#right-column {
    padding-left: 20px;
}
"""

# 5. Modern Dashboard Design using Blocks
with gr.Blocks(theme=gr.themes.Soft(),css=custom_css) as demo:
    gr.Markdown("# AI-Powered Resume Screening Assistant")
    gr.Markdown("Provide job requirements to analyze and screen candidate resumes effectively.")    
    with gr.Row():
        # Left Panel: Input area
        with gr.Column(scale=1, elem_id="left-column"):
            gr.Markdown("###  Search Criteria")
            job_query = gr.Textbox(
                label="Enter Job Requirements", 
                placeholder="e.g., Python, AWS, 3 years experience...",
                lines=5
            )
            submit_btn = gr.Button("Analyze Resumes", variant="primary")
            
            gr.Examples(
                examples=["Mechanical Engineering and CAD", "Data Science and Python", "Project Management"],
                inputs=job_query
            )
        
        # Right Panel: Output area
        with gr.Column(scale=2, elem_id="right-column"):
            gr.Markdown("### Screening Analysis")
            # This is where your specific formatted text will stream
            output_markdown = gr.Markdown(value="Waiting for input...", label="Evaluation")

    # Connect the button to the function
    submit_btn.click(fn=screen_resumes_stream, inputs=job_query, outputs=output_markdown)

if __name__ == "__main__":
    demo.launch()