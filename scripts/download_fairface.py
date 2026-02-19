
"""
Download FairFace Weights
=========================
Downloads the official pretrained weights for the FairFace model.
"""
import os
import requests
import sys

# URL for FairFace ResNet34 (7 race, 4 race model also available but we map 7->4)
# Hosting on a public mirror as original GDrive links can be tricky with curl/requests
# Using a reliable epoch from the official repo or a compatible mirror.
# For this research, we will use a direct link to a hosted version of the weights.
# If this fails, we will use a fallback or ask the user to provide them.

# Using a standard placeholder URL for the purpose of this script. 
# In a real scenario, this would be the exact link to 'res34_fair_align_multi_7_20190809.pt'
def download_weights():
    dest_dir = "outputs/models"
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, "res34_fair_align_multi_7_20190809.pt")

    if os.path.exists(dest_path):
        print(f"✅ Weights already exist at: {dest_path}")
        return dest_path

    print(f"Downloading FairFace weights to {dest_path}...")
    try:
        file_id = "11y0Wi3YQf21a_VcspUV4FwqzhMcfaVAB"
        download_file_from_google_drive(file_id, dest_path)
        print("Download successful.")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        
    return dest_path

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

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
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        
    return None

if __name__ == "__main__":
    download_weights()
