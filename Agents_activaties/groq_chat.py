import gradio as gr
from groq import Groq
import os

key = os.getenv("groq")

client = Groq(
    api_key=key)


def chat(message,history):
  chat_completion = client.chat.completions.create(
      messages=[
          {
              "role": "system",
              "content": "You are a helpful assistant.",
          },
                
          {
              "role": "user",
              "content": message,
          }
      ],
      model="llama3-8b-8192",
      max_tokens=256,
      temperature=0.7,
      top_p=1,
      stop = None,
      stream = False,
  )

  return chat_completion.choices[0].message.content

demo = gr.ChatInterface(fn = chat, title = "Open Source Chatbot")

demo.launch(debug= True)