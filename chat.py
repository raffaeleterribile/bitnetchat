""" Demonstrates a simple chat interface. """
import gradio as gr
from ai import generate

examples = [
	["""Daniel went back to the garden. Mary travelled to the kitchen. Sandra journeyed to the kitchen. Sandra went to the hallway. John went to the bedroom. Mary went back to the garden. Where is Mary?\nAnswer: """],
]

demo = gr.ChatInterface(generate, type="messages", examples=examples)

demo.launch()
