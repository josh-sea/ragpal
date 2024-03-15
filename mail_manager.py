import json
import base64
from bs4 import BeautifulSoup
from json.decoder import JSONDecodeError
from datetime import datetime, timezone
from auth import *

def get_current_epoch_local():
    """Return the current local time as an epoch timestamp in seconds."""
    return int(datetime.now().timestamp())


def save_last_query_timestamp():    
    # Save to a local JSON file
    timestamp = get_current_epoch_local()
    with open('last_query_timestamp.json', 'w') as file:
        json.dump({'last_query_timestamp': timestamp}, file)

def fetch_latest_emails(after=None, max_results=25,  user_id='me',):
    service = gmail_authenticate()  # Your Gmail authentication function
    save_last_query_timestamp()
    emails = []
    query = "category:primary"
    if after:
        query += f" after:{after}"
        results = service.users().messages().list(userId=user_id, q=query, maxResults=max_results).execute()
        messages = results.get('messages', [])
        for message in messages:
            msg = service.users().messages().get(userId=user_id, id=message['id']).execute()
            emails.append(msg)
    else:
        try:
            with open('last_query_timestamp.json', 'r') as file:
                last_run = json.load(file)
                
            query += f" after:{last_run.last_query_timestamp}"
            results = service.users().messages().list(userId=user_id, q=query, maxResults=max_results).execute()
            messages = results.get('messages', [])
            for message in messages:
                msg = service.users().messages().get(userId=user_id, id=message['id']).execute()
                emails.append(msg)
        except JSONDecodeError:
            # Return an empty dictionary if the JSON is invalid or the file is empty
            return {}
        except FileNotFoundError:
            # Return an empty dictionary if the file does not exist
            return {}
    processed_emails = [process_message(email) for email in emails]  # Process each email
    return processed_emails

def load_existing_messages(filename='messages.json'):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except JSONDecodeError:
        # Return an empty dictionary if the JSON is invalid or the file is empty
        return {}
    except FileNotFoundError:
        # Return an empty dictionary if the file does not exist
        return {}


def save_messages(messages, filename='messages.json'):
    with open(filename, 'w') as file:
        json.dump(messages, file, indent=2)

def append_new_messages(new_emails, existing_messages):
    updated = False
    for email in new_emails:
        message_id = email['id']
        if message_id not in existing_messages:
            existing_messages[message_id] = email
            updated = True
    return existing_messages, updated

def process_message(message):
    processed = {
        "id": message["id"],
        "threadId": message["threadId"],
        "labelIds": message.get("labelIds", []),
        "snippet": message.get("snippet", ""),
        "subject": "",
        "from": "",
        "to": "",
        "date": "",
        "content": "",
    }

    # Extract headers
    for header in message["payload"]["headers"]:
        name = header["name"].lower()
        if name == "from":
            processed["from"] = header["value"]
            print(f'from {header["value"]}')
        elif name == "to":
            processed["to"] = header["value"]
        elif name == "subject":
            processed["subject"] = header["value"]
            print(f'from {header["value"]}')
        elif name == "date":
            processed["date"] = header["value"]

    # Extract and decode body
    parts = message["payload"].get("parts", [])
    if not parts:  # For non-multipart messages
        parts = [message["payload"]]

    for part in parts:
        if part["mimeType"] == "text/plain" or part["mimeType"] == "text/html":
            data = part["body"].get("data", "")
            decoded_data = str(base64.urlsafe_b64decode(data), "utf-8")
            # Optionally clean HTML content
            if part["mimeType"] == "text/html":
                soup = BeautifulSoup(decoded_data, "html.parser")
                processed["content"] += soup.get_text(separator="\n")
            else:
                processed["content"] += decoded_data

    return processed