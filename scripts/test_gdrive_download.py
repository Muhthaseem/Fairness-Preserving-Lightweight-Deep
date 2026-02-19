
import requests
import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    file_id = '11y0Wi3YQf21a_VcspUV4FwqzhMcfaVAB'
    destination = 'res34_fair_align_multi_7_20190809.pt'
    print(f"Attempting download of ID {file_id} to {destination}...")
    try:
        download_file_from_google_drive(file_id, destination)
        if os.path.exists(destination):
            size = os.path.getsize(destination)
            print(f"Download complete. Size: {size/1024/1024:.2f} MB")
        else:
             print("Download failed: File not created.")
    except Exception as e:
        print(f"Error: {e}")
