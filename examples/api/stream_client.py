import requests
import json
import os
from tools.audio import pcm_arr_to_mp3_view

# Define the endpoint URL
API_URL = "http://0.0.0.0:8001/generate_voice_stream"  # Replace with your actual server URL

# Define the payload matching ChatTTSParams
# Adjust the structure according to your actual ChatTTSParams model
body = {
    "text": [
        "四川美食确实以辣闻名，但也有不辣的选择。",
        "比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。",
    ],
    "stream": True,
    "lang": None,
    "skip_refine_text": True,
    "refine_text_only": False,
    "use_decoder": True,
    "audio_seed": 12345678,
    "text_seed": 87654321,
    "do_text_normalization": True,
    "do_homophone_replacement": False,
}

# refine text params
params_refine_text = {
    "prompt": "",
    "top_P": 0.7,
    "top_K": 20,
    "temperature": 0.7,
    "repetition_penalty": 1,
    "max_new_token": 384,
    "min_new_token": 0,
    "show_tqdm": True,
    "ensure_non_empty": True,
    "stream_batch": 24,
}
body["params_refine_text"] = params_refine_text

# infer code params
params_infer_code = {
    "prompt": "[speed_5]",
    "top_P": 0.1,
    "top_K": 20,
    "temperature": 0.3,
    "repetition_penalty": 1.05,
    "max_new_token": 2048,
    "min_new_token": 0,
    "show_tqdm": True,
    "ensure_non_empty": True,
    "stream_batch": True,
    "spk_emb": None,
}
body["params_infer_code"] = params_infer_code

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
        with requests.post(api_url, json=body, stream=True) as response:
            response.raise_for_status()  # Raise an error for bad status codes

            # Check if the response is a WAV stream
            content_type = response.headers.get('Content-Type', '')
            if 'audio/wav' not in content_type:
                print(f"Unexpected Content-Type: {content_type}")
                print("Response Text:", response.text)
                return

            # Open the output file in binary write mode
            with open('output.wav', 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
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
    test_generate_voice_streaming(API_URL, body, output_wav)
