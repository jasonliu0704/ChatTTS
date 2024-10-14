import io
import os
import sys
import zipfile
import wave
import numpy as np

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import soundfile as sf


if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

from typing import Optional

import ChatTTS

from tools.audio import pcm_arr_to_mp3_view
from tools.logger import get_logger
# from tools.audio.pcm import pcm_arr_to_wav_bytes

import torch


from pydantic import BaseModel


logger = get_logger("Command")

app = FastAPI()

logger.info("loading chat")
chat = ChatTTS.Chat(get_logger("ChatTTS"))
logger.info("Initializing ChatTTS...")
if chat.load():
    logger.info("Models loaded successfully.")
else:
    logger.error("Models load failed.")
    sys.exit(1)
logger.info("Done loading ChatTTS")

# @app.on_event("startup")
# async def startup_event():
#     global chat

#     chat = ChatTTS.Chat(get_logger("ChatTTS"))
#     logger.info("Initializing ChatTTS...")
#     if chat.load():
#         logger.info("Models loaded successfully.")
#     else:
#         logger.error("Models load failed.")
#         sys.exit(1)


class ChatTTSParams(BaseModel):
    text: list[str]
    stream: bool = False
    lang: Optional[str] = None
    skip_refine_text: bool = False
    refine_text_only: bool = False
    use_decoder: bool = True
    do_text_normalization: bool = True
    do_homophone_replacement: bool = False
    params_refine_text: ChatTTS.Chat.RefineTextParams
    params_infer_code: ChatTTS.Chat.InferCodeParams


app = FastAPI()

@app.post("/generate_voice_stream")
async def generate_voice_stream(params: ChatTTSParams):
    logger.info("Text input: %s", str(params.text))

    # ... (Your existing code for inference and processing)

    logger.info("Start voice inference.")
    wavs = chat.infer(
        text=params.text,
        stream=False, #params.stream,
        lang=params.lang,
        skip_refine_text=params.skip_refine_text,
        use_decoder=params.use_decoder,
        do_text_normalization=params.do_text_normalization,
        do_homophone_replacement=params.do_homophone_replacement,
        params_infer_code=params.params_infer_code,
        params_refine_text=params.params_refine_text,
    )
    logger.info("Inference completed.")

    # Assuming 'wavs' is a list of NumPy arrays representing audio data
    # If you have multiple wavs, you can concatenate them
    import numpy as np
    combined_wav = np.concatenate(wavs)

    # Convert the NumPy array to bytes in WAV format
    buf = io.BytesIO()
    import soundfile as sf

    # Write the WAV data to the buffer
    sf.write(buf, combined_wav, samplerate=24000, format='WAV')
    buf.seek(0)

    response = StreamingResponse(buf, media_type="audio/wav")
    response.headers["Content-Disposition"] = "attachment; filename=audio.wav"
    return response

# def pcm_arr_to_pcm_bytes(pcm_data: np.ndarray, bit_depth: int) -> bytes:
#     # Define the maximum and minimum amplitude based on bit depth
#     if bit_depth == 8:
#         dtype = np.uint8
#         max_amplitude = np.iinfo(np.uint8).max  # 255
#         min_amplitude = np.iinfo(np.uint8).min  # 0
#         # Scale floating-point data from [-1.0, 1.0] to [0, 255]
#         pcm_data = np.clip(pcm_data, -1.0, 1.0)
#         pcm_data = ((pcm_data + 1.0) * 127.5).astype(dtype)
#     else:
#         if bit_depth == 16:
#             dtype = np.int16
#             max_amplitude = np.iinfo(np.int16).max   # 32767
#             min_amplitude = np.iinfo(np.int16).min   # -32768
#         elif bit_depth == 24:
#             dtype = np.int32  # We'll handle 24-bit manually
#             max_amplitude = 8388607
#             min_amplitude = -8388608
#         elif bit_depth == 32:
#             dtype = np.int32
#             max_amplitude = np.iinfo(np.int32).max   # 2147483647
#             min_amplitude = np.iinfo(np.int32).min   # -2147483648

#         # Clip floating-point data to [-1.0, 1.0]
#         pcm_data = np.clip(pcm_data, -1.0, 1.0)

#         # Scale to integer range
#         pcm_data = (pcm_data * max_amplitude).astype(dtype)

#         # Clip integer PCM data to valid range [min_amplitude, max_amplitude]
#         pcm_data = np.clip(pcm_data, min_amplitude, max_amplitude)

#     # Handle 24-bit data separately
#     if bit_depth == 24:
#         pcm_bytes = bytearray()
#         for sample in pcm_data:
#             # Convert each sample to bytes and take the first 3 bytes (little endian)
#             sample_bytes = sample.to_bytes(4, byteorder='little', signed=True)
#             pcm_bytes.extend(sample_bytes[:3])  # Take the least significant 3 bytes
#         return bytes(pcm_bytes)
#     else:
#         return pcm_data.tobytes()


import wave

def pcm_arr_to_wav_bytes(
    pcm_data: np.ndarray,
    sample_rate: int = 24000,
    num_channels: int = 1,
    bit_depth: int = 16
) -> bytes:
    sampwidth = bit_depth // 8

    # Create a BytesIO buffer to hold the WAV data
    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(sample_rate)
        # Write empty frames to generate the header
        wav_file.writeframes(b'')

    wav_bytes = wav_buffer.getvalue()
    wav_buffer.close()

    return wav_bytes

def pcm_arr_to_pcm_bytes(
    pcm_data: np.ndarray,
    bit_depth: int = 16
) -> bytes:
    # Determine the appropriate data type and scaling
    if bit_depth == 8:
        dtype = np.uint8  # Unsigned 8-bit
        max_amplitude = np.iinfo(np.uint8).max  # 255
        # Scale from [-1.0, 1.0] to [0, 255]
        pcm_data = np.clip(pcm_data, -1.0, 1.0)
        pcm_data = ((pcm_data + 1.0) * 127.5).astype(dtype)
    else:
        if bit_depth == 16:
            dtype = np.dtype('<i2')  # Little-endian 16-bit signed integer
            max_amplitude = np.iinfo(np.int16).max  # 32767
            min_amplitude = np.iinfo(np.int16).min  # -32768
        elif bit_depth == 24:
            dtype = np.int32  # Will handle manually
            max_amplitude = 8388607
            min_amplitude = -8388608
        elif bit_depth == 32:
            dtype = np.dtype('<i4')  # Little-endian 32-bit signed integer
            max_amplitude = np.iinfo(np.int32).max  # 2147483647
            min_amplitude = np.iinfo(np.int32).min  # -2147483648

        # Clip floating-point data to [-1.0, 1.0]
        pcm_data = np.clip(pcm_data, -1.0, 1.0)

        # Scale to integer range and ensure data type
        pcm_data = (pcm_data * max_amplitude).astype(dtype)
        pcm_data = np.clip(pcm_data, min_amplitude, max_amplitude)

    if bit_depth == 24:
        # Manually handle 24-bit data
        pcm_bytes = bytearray()
        for sample in pcm_data:
            sample_bytes = sample.astype(np.int32).tobytes()
            pcm_bytes.extend(sample_bytes[:3])  # Take least significant 3 bytes
        return bytes(pcm_bytes)
    else:
        return pcm_data.tobytes()

@app.post("/generate_voice_stream_live")
async def generate_voice_stream_live(params: ChatTTSParams):
    logger.info("Text input: %s", str(params.text))

    # Audio seed
    if params.params_infer_code.manual_seed is not None:
        torch.manual_seed(params.params_infer_code.manual_seed)
        params.params_infer_code.spk_emb = chat.sample_random_speaker()

    # Text seed for text refining
    if params.params_refine_text:
        text = chat.infer(
            text=params.text, skip_refine_text=False, refine_text_only=True
        )
        logger.info(f"Refined text: {text}")
    else:
        # No text refining
        text = params.text

    logger.info("Using speaker:")
    logger.info(params.params_infer_code.spk_emb)

    logger.info("Starting voice inference.")

    # Assume chat.infer() is modified to be a generator that yields PCM data
    wavs = chat.infer(
        text=text,
        stream=params.stream,
        lang=params.lang,
        skip_refine_text=params.skip_refine_text,
        use_decoder=params.use_decoder,
        do_text_normalization=params.do_text_normalization,
        do_homophone_replacement=params.do_homophone_replacement,
        params_infer_code=params.params_infer_code,
        params_refine_text=params.params_refine_text,
    )

    # Prepare to save the audio to a file
    output_file_path = 'output.wav'  # You can customize this path
    sample_rate = 24000  # Ensure this matches your PCM data's sample rate
    num_channels = 1
    bit_depth = 16  # Ensure this matches your PCM data's bit depth

    # Open the file for writing in binary mode
    output_file = open(output_file_path, 'wb')

    def stream_wav():
        # Access output_file from the enclosing scope
        # Prepare the WAV header
        wav_header = pcm_arr_to_wav_bytes(
            pcm_data=np.array([], dtype=np.float32),
            sample_rate=sample_rate,
            num_channels=num_channels,
            bit_depth=bit_depth
        )

        # Write the WAV header to the file
        output_file.write(wav_header)

        # Yield the WAV header to the client
        yield wav_header

        try:
            # Now stream the PCM data
            for wav in wavs:
                for w in wav:
                    # Ensure PCM data is within [-1.0, 1.0]
                    w = np.clip(w, -1.0, 1.0)

                    # Convert PCM array to bytes (without header)
                    pcm_bytes = pcm_arr_to_pcm_bytes(w, bit_depth=bit_depth)

                    # Write the PCM bytes to the file
                    output_file.write(pcm_bytes)

                    # Yield the PCM bytes to the client
                    yield pcm_bytes
        finally:
            # Close the file when done
            output_file.close()

    logger.info("Inference started. Streaming WAV data to client.")

    # Stream the WAV data using StreamingResponse
    return StreamingResponse(stream_wav(), media_type="audio/wav")



@app.post("/generate_voice")
async def generate_voice(params: ChatTTSParams):
    logger.info("Text input: %s", str(params.text))

    # audio seed
    if params.params_infer_code.manual_seed is not None:
        torch.manual_seed(params.params_infer_code.manual_seed)
        params.params_infer_code.spk_emb = chat.sample_random_speaker()

    # text seed for text refining
    if params.params_refine_text:
        text = chat.infer(
            text=params.text, skip_refine_text=False, refine_text_only=True
        )
        logger.info(f"Refined text: {text}")
    else:
        # no text refining
        text = params.text

    logger.info("Use speaker:")
    logger.info(params.params_infer_code.spk_emb)

    logger.info("Start voice inference.")
    wavs = chat.infer(
        text=text,
        stream=params.stream,
        lang=params.lang,
        skip_refine_text=params.skip_refine_text,
        use_decoder=params.use_decoder,
        do_text_normalization=params.do_text_normalization,
        do_homophone_replacement=params.do_homophone_replacement,
        params_infer_code=params.params_infer_code,
        params_refine_text=params.params_refine_text,
    )
    logger.info("Inference completed.")

    # zip all of the audio files together
    buf = io.BytesIO()
    with zipfile.ZipFile(
        buf, "a", compression=zipfile.ZIP_DEFLATED, allowZip64=False
    ) as f:
        for idx, wav in enumerate(wavs):
            f.writestr(f"{idx}.mp3", pcm_arr_to_mp3_view(wav))
    logger.info("Audio generation successful.")
    buf.seek(0)

    response = StreamingResponse(buf, media_type="application/zip")
    response.headers["Content-Disposition"] = "attachment; filename=audio_files.zip"
    return response


# uvicorn  examples.api.main:app --host 0.0.0.0 --port 8001