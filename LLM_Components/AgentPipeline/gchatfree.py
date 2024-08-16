import g4f
import sys
import asyncio
# Set the appropriate event loop policy for Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

response = g4f.ChatCompletion.create(model='gpt-3.5-turbo', provider=g4f.Provider.You,
                                    messages=[{"role": "user", "content": "Hello world"}], stream=g4f.Provider.You.supports_stream)


for message in response:
    # print(message)
    print(message, flush=True, end='')


from g4f.cookies import set_cookies

set_cookies(".bing.com", {
  "_U": "cookie value"
})

set_cookies(".google.com", {
  "__Secure-1PSID": "cookie value"
})
# from g4f.client import Client

# client = Client()
# response = client.images.generate(
#   model="gemini",
#   prompt="a white siamese cat",
 
# )
# image_url = response.data[0].url

