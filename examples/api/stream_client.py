import requests
import json
import os
from tools.audio import pcm_arr_to_mp3_view

from tools.audio import float_to_int16

import requests
import io
import wave

# Define the endpoint URL
API_URL = "http://0.0.0.0:8001/generate_voice_chat_stream"  # Updated endpoint URL

# Define the payload matching ChatTTSParams
body = {
    "text": [
        "四川美食确实以辣闻名，但也有不辣的选择。",
        "比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。",
    ],
    "stream": True,  # Set to True to enable streaming
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
    try:
        with requests.post(api_url, json=payload, stream=True) as response:
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '')
            if 'audio/wav' not in content_type.lower():
                print(f"Unexpected Content-Type: {content_type}")
                print("Response Text:", response.text)
                return

            # Read the initial WAV header
            header = response.raw.read(44)  # WAV header is 44 bytes
            data_chunks = []

            # Read the rest of the data
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    data_chunks.append(chunk)

            # Combine all data chunks
            audio_data = b''.join(data_chunks)

            # Calculate the correct data sizes
            data_size = len(audio_data)
            chunk_size = data_size + 36  # 36 bytes for header excluding 'RIFF' and 'WAVE' identifiers

            # Reconstruct the WAV header with correct sizes
            header = bytearray(header)
            header[4:8] = chunk_size.to_bytes(4, 'little')       # ChunkSize
            header[40:44] = data_size.to_bytes(4, 'little')      # Subchunk2Size

            # Write the corrected WAV file
            with open(output_file_path, 'wb') as f:
                f.write(header)
                f.write(audio_data)

            print(f"Streaming completed. WAV file saved as {output_file_path}.")

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")  # HTTP error
    except Exception as err:
        print(f"An error occurred: {err}")          # Other errors



if __name__ == "__main__":
    # Define the output file path
    output_wav = "test_output.wav"

    # Optionally, remove the file if it already exists
    if os.path.exists(output_wav):
        os.remove(output_wav)

    # Call the test function
    test_generate_voice_streaming(API_URL, body, output_wav)
