import requests
import json
import os

# Define the endpoint URL
API_URL = "http://0.0.0.0:8000/generate_voice_stream"  # Replace with your actual server URL

# Define the payload matching ChatTTSParams
# Adjust the structure according to your actual ChatTTSParams model
payload = {
    "text": "Hello, this is a test of the text-to-speech streaming.",
    "params_infer_code": {
        "manual_seed": 42,  # Example seed value; set to None if not using
        "spk_emb": None      # This will be set by the server if manual_seed is provided
    },
    "params_refine_text": True,
    "stream": True,
    "lang": "en",
    "skip_refine_text": False,
    "use_decoder": True,
    "do_text_normalization": True,
    "do_homophone_replacement": False,
    # Add other necessary fields as per ChatTTSParams
}

def test_generate_voice_streaming(api_url, payload, output_file_path):
    """
    Sends a POST request to the generate_voice endpoint and streams the WAV response to a file.

    Parameters:
    - api_url (str): The URL of the FastAPI endpoint.
    - payload (dict): The JSON payload to send in the POST request.
    - output_file_path (str): The path where the streamed WAV file will be saved.
    """
    try:
        # Send the POST request with streaming enabled
        with requests.post(api_url, json=payload, stream=True) as response:
            response.raise_for_status()  # Raise an error for bad status codes

            # Check if the response is a WAV stream
            content_type = response.headers.get('Content-Type', '')
            if 'audio/wav' not in content_type:
                print(f"Unexpected Content-Type: {content_type}")
                print("Response Text:", response.text)
                return

            # Open the output file in binary write mode
            with open(output_file_path, 'wb') as f:
                print(f"Streaming WAV data to {output_file_path}...")
                
                # Iterate over the response in chunks
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)

            print(f"Streaming completed. WAV file saved as {output_file_path}.")

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")  # HTTP error
    except Exception as err:
        print(f"An error occurred: {err}")          # Other errors

if __name__ == "__main__":
    # Define the output file path
    output_wav = "output.wav"

    # Optionally, remove the file if it already exists
    if os.path.exists(output_wav):
        os.remove(output_wav)

    # Call the test function
    test_generate_voice_streaming(API_URL, payload, output_wav)
