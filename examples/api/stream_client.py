import wave
import numpy as np
import requests
import json
import os
from tools.audio import pcm_arr_to_mp3_view

from tools.audio import float_to_int16

import requests

# Define the endpoint URL
API_URL = "http://0.0.0.0:8001/generate_voice_chat_stream"  # Updated endpoint URL

# Define the payload matching ChatTTSParams
body = {
    "text": [
        "This is real shit hey hey hey",
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

            # Open the output file in binary write mode
            with open(output_file_path, 'wb') as f:
                # Initialize variables to keep track of the header and data
                header = b''
                data_size = 0

                # Create an iterator for the response content
                chunks = response.iter_content(chunk_size=16384)

                # Read the WAV header (44 bytes)
                while len(header) < 44:
                    try:
                        chunk = next(chunks)
                    except StopIteration:
                        raise ValueError("Received incomplete WAV header.")

                    if chunk:
                        header += chunk

                # Split the header and the initial data
                wav_header = header[:44]
                remaining_header = header[44:]

                # Write the header to the file
                f.write(wav_header)

                # Write any remaining data from the header chunk
                if remaining_header:
                    f.write(remaining_header)
                    data_size += len(remaining_header)

                # Write the rest of the audio data to the file as it is received
                chunk_counter = 0
                for chunk in chunks:
                    if chunk:
                        f.write(chunk)
                        data_size += len(chunk)
                        
                        # Save each chunk to a separate file
                        chunk_file_path = f"{output_file_path}_chunk_{chunk_counter}.wav"
                        with wave.open(chunk_file_path, 'wb') as chunk_file:
                            chunk_file.setnchannels(1)  # Mono audio
                            chunk_file.setsampwidth(2)  # 2 bytes per sample (16-bit audio)
                            chunk_file.setframerate(16000)  # Assuming a sample rate of 16kHz
                            chunk_file.writeframes(chunk)
                        chunk_counter += 1

            # After writing all data, update the header with correct sizes
            chunk_size = data_size + 36  # 36 bytes for header excluding 'RIFF' and 'WAVE' identifiers

            # Open the file in read-write binary mode to update the header
            with open(output_file_path, 'r+b') as f:
                f.seek(4)  # Move to ChunkSize position
                f.write(chunk_size.to_bytes(4, 'little'))
                f.seek(40)  # Move to Subchunk2Size position
                f.write(data_size.to_bytes(4, 'little'))

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
