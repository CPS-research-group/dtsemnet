"""For sending discord messages."""
import requests
def send_discord_message(content, WEBHOOK_URL):
    """Send a message to the Discord channel."""
    data = {"content": content}
    try:
        response = requests.post(WEBHOOK_URL, json=data)
        response.raise_for_status()  # Raise an error for bad status codes
    except requests.exceptions.RequestException as e:
        print(f"Failed to send Discord message: {e}")
