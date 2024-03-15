import requests
import os
from elevenlabs.client import ElevenLabs
from elevenlabs import play, Voice

client = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_API_KEY")
)
voice = "sFGsuVsnf5ieYfUlwn35"
url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
# text = "The bill directs the Under Secretary for Management of DHS to assess contracts for covered services along the U.S.-Mexico border in the following ways:  1. Submit a report to Congress within 180 days of enactment regarding active DHS contracts for covered border services awarded on or before September 30, 2023 or the date of enactment, whichever is later.   2. The report must include the criteria DHS used to determine the necessity of contractor personnel in assisting with the covered services.  The bill text provided does not mention any budgets that must be proposed, public meetings that must be held, or standard processes that are bypassed in order to support this legislation.   It also does not specify any particular areas along the southern border that need to be addressed. The assessment appears to cover all active contracts for covered services performed by contractors along the entire U.S.-Mexico border."
text = "hello!"
payload = {
    "model_id": "eleven_multilingual_v2",
    "text": text
}

headers = {
    "xi-api-key": os.getenv("ELEVENLABS_API_KEY"),
    "Content-Type": "application/json"
}

response = requests.request("POST", url, json=payload, headers=headers)
print(response)
print(response.text)

# ate
# Thu, 14 Mar 2024 00:14:48 GMT
# server
# uvicorn
# request-id
# 8sN8kX6YG3vn27U2Xdpm
# history-item-id
# d1KBxoFWoCUboHTHX31l
# access-control-expose-headers
# request-id, history-item-id, tts-latency-ms
# tts-latency-ms
# 13634
# content-length
# 357355
# content-type
# audio/mpeg
# access-control-allow-origin
# *
# access-control-allow-headers
# *
# access-control-allow-methods
# POST, OPTIONS, DELETE, GET, PUT
# access-control-max-age
# 600
# strict-transport-security
# max-age=31536000; includeSubDomains
# via
# 1.1 google
# alt-svc
# h3=":443"; ma=2592000,h3-29=":443"; ma=2592000
# connection
# close
# # jarvis (jbone)
# audio = client.generate(
#   text=text,
#   voice=Voice(voice_id=voice)
# )
# play(audio) 