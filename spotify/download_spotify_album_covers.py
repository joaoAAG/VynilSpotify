import os
import requests
from PIL import Image
from io import BytesIO
from requests.auth import HTTPBasicAuth

# Spotify API credentials
CLIENT_ID = '993bd62e07754408b9aac7cf93444259'
CLIENT_SECRET = '03be7022eb4d43fc820522ef2bd6ee5c'

# Directory to save album covers
album_covers_dir = 'spotify_album_covers'
os.makedirs(album_covers_dir, exist_ok=True)


# Function to get Spotify access token
def get_spotify_token():
    url = 'https://accounts.spotify.com/api/token'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {
        'grant_type': 'client_credentials'
    }
    response = requests.post(url, headers=headers, data=data, auth=HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET))
    token = response.json().get('access_token')
    return token


# Function to search for albums and download covers
def search_and_download_album_covers(album_name, num_images=10):
    token = get_spotify_token()
    headers = {
        'Authorization': f'Bearer {token}'
    }
    search_url = f'https://api.spotify.com/v1/search?q={album_name}&type=album&limit={num_images}'
    search_response = requests.get(search_url, headers=headers).json()

    if 'albums' not in search_response or 'items' not in search_response['albums']:
        print(f"No albums found for query: {album_name}")
        return

    albums = search_response['albums']['items']
    for i, album in enumerate(albums):
        album_images = album['images']
        if album_images:
            image_url = album_images[0]['url']
            try:
                img_response = requests.get(image_url)
                img = Image.open(BytesIO(img_response.content))
                sanitized_album_name = album_name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":",
                                                                                                                 "_")
                img.save(os.path.join(album_covers_dir, f'spotify_album_{sanitized_album_name}_{i}.jpg'))
                print(f'Downloaded: spotify_album_{sanitized_album_name}_{i}.jpg')
            except Exception as e:
                print(f"Could not download image {i} from {image_url}. Error: {e}")


# Example usage
album_names = ["Michael Jackson - Bad", "AC/DC - Highway to Hell", "AC/DC - High Voltage"]
for album_name in album_names:
    search_and_download_album_covers(album_name, num_images=1)

print("Album covers downloaded successfully.")
