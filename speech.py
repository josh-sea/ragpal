import requests
# jarvis (jbone)
voice_id= "sFGsuVsnf5ieYfUlwn35"
api_key = "0674d03bef25b44f2c1816d72de268f1"
url = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

import requests

url = "https://api.elevenlabs.io/v1/text-to-speech/sFGsuVsnf5ieYfUlwn35"

payload = {"text": "The bill directs the Under Secretary for Management of DHS to assess contracts for covered services along the U.S.-Mexico border in the following ways:  1. Submit a report to Congress within 180 days of enactment regarding active DHS contracts for covered border services awarded on or before September 30, 2023 or the date of enactment, whichever is later.   2. The report must include the criteria DHS used to determine the necessity of contractor personnel in assisting with the covered services.  The bill text provided does not mention any budgets that must be proposed, public meetings that must be held, or standard processes that are bypassed in order to support this legislation.   It also does not specify any particular areas along the southern border that need to be addressed. The assessment appears to cover all active contracts for covered services performed by contractors along the entire U.S.-Mexico border."}
headers = {
    "xi-api-key": api_key,
    "Content-Type": "application/json"
}

response = requests.request("POST", url, json=payload, headers=headers)

print(response.text)
