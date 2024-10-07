import io
import os
import sys
import zipfile
import wave
import numpy as np

from fastapi import FastAPI
from fastapi.responses import StreamingResponse


if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

from typing import Optional

import ChatTTS

from tools.audio import pcm_arr_to_mp3_view
from tools.logger import get_logger
from tools.audio.pcm import pcm_arr_to_wav_bytes

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
logger.info("Done loading the chat")

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
import soundfile as sf

@app.post("/generate_voice_stream")
async def generate_voice_stream(params: ChatTTSParams):
    logger.info("Text input: %s", str(params.text))

    # ... (Your existing code for inference and processing)

    logger.info("Start voice inference.")
    wavs = chat.infer(
        text=params.text,
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

    def stream_wav():
        for wav in wavs:

            # Assuming 'wavs' is a list of NumPy arrays representing audio data
            # If you have multiple wavs, you can concatenate them
            # import numpy as np
            # combined_wav = np.concatenate(wavs)

            # Convert the NumPy array to bytes in WAV format
            buf = io.BytesIO()

            # Write the WAV data to the buffer
            sf.write(buf, wav, samplerate=24000, format='WAV')
            buf.seek(0)
            yield buf

    response = StreamingResponse(stream_wav(), media_type="audio/wav")
    response.headers["Content-Disposition"] = "attachment; filename=audio.wav"
    return response

# async def generate_voice_stream(params: ChatTTSParams):
#     logger.info("Text input: %s", str(params.text))

#     # Audio seed
#     if params.params_infer_code.manual_seed is not None:
#         torch.manual_seed(params.params_infer_code.manual_seed)
#         params.params_infer_code.spk_emb = chat.sample_random_speaker()

#     # Text seed for text refining
#     if params.params_refine_text:
#         text = chat.infer(
#             text=params.text, skip_refine_text=False, refine_text_only=True
#         )
#         logger.info(f"Refined text: {text}")
#     else:
#         # No text refining
#         text = params.text

#     logger.info("Using speaker:")
#     logger.info(params.params_infer_code.spk_emb)

#     logger.info("Starting voice inference.")

#     # Assume chat.infer() is modified to be a generator that yields WAV data
#     wavs = chat.infer(
#         text=text,
#         stream=params.stream,
#         lang=params.lang,
#         skip_refine_text=params.skip_refine_text,
#         use_decoder=params.use_decoder,
#         do_text_normalization=params.do_text_normalization,
#         do_homophone_replacement=params.do_homophone_replacement,
#         params_infer_code=params.params_infer_code,
#         params_refine_text=params.params_refine_text,
#     )

#     # Generator function to process and yield WAV data
#     def stream_wav():
#         idx = 0
#         for wav in wavs:
#             logger.info(f"Processing WAV {idx}")
#             for i, w in enumerate(wav):
#                 # Convert PCM array to WAV bytes
#                 wav_bytes = pcm_arr_to_wav_bytes(w, bit_depth=-1)
#                 yield wav_bytes
#             idx += 1

#     logger.info("Inference started. Streaming WAV data to client.")

#     # Stream the WAV data using StreamingResponse
#     return StreamingResponse(stream_wav(), media_type="audio/wav")

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