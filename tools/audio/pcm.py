import wave
from io import BytesIO

import numpy as np
import wave

from .np import float_to_int16
from .av import wav2


def pcm_arr_to_mp3_view(wav: np.ndarray):
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # Sample width in bytes
        wf.setframerate(24000)  # Sample rate in Hz
        wf.writeframes(float_to_int16(wav))
    buf.seek(0, 0)
    buf2 = BytesIO()
    wav2(buf, buf2, "mp3")
    buf.seek(0, 0)
    return buf2.getbuffer()

def pcm_arr_to_wav_bytes(
    pcm_data: np.ndarray,
    sample_rate: int = 24000,
    num_channels: int = 1,
    bit_depth: int = 16
) -> bytes:
    # Validate bit depth
    if bit_depth not in (8, 16, 24, 32):
        raise ValueError("Unsupported bit depth. Choose from 8, 16, 24, or 32 bits.")

    # Determine the appropriate WAV format
    if bit_depth == 8:
        sampwidth = 1  # 1 byte
        dtype = np.uint8  # WAV 8-bit is unsigned
        max_amplitude = np.iinfo(np.uint8).max
    elif bit_depth == 16:
        sampwidth = 2  # 2 bytes
        dtype = np.int16
        max_amplitude = np.iinfo(np.int16).max
    elif bit_depth == 24:
        sampwidth = 3  # 3 bytes
        dtype = np.int32  # We'll handle 24-bit manually
        max_amplitude = 8388607  # Max value for 24-bit audio
    elif bit_depth == 32:
        sampwidth = 4  # 4 bytes
        dtype = np.int32
        max_amplitude = np.iinfo(np.int32).max

    # Scale floating-point PCM data to integer PCM data
    if np.issubdtype(pcm_data.dtype, np.floating):
        # Ensure data is in the range [-1.0, 1.0]
        pcm_data = np.clip(pcm_data, -1.0, 1.0)
        pcm_data = (pcm_data * max_amplitude).astype(dtype)
    else:
        pcm_data = pcm_data.astype(dtype)

    # Create a BytesIO buffer to hold the WAV data
    wav_buffer = BytesIO()

    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(sample_rate)

        if bit_depth == 24:
            # Handle 24-bit data manually
            pcm_bytes = pcm_data.tobytes()
            pcm_bytes_24 = bytearray()
            for i in range(0, len(pcm_bytes), 4):
                pcm_bytes_24.extend(pcm_bytes[i:i+3])
            wav_file.writeframes(pcm_bytes_24)
        else:
            wav_file.writeframes(pcm_data.tobytes())

    wav_bytes = wav_buffer.getvalue()
    wav_buffer.close()

    return wav_bytes


# Generate a 1-second 440 Hz sine wave as an example
def generate_sine_wave(freq, duration, sample_rate=44100, amplitude=32767):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = amplitude * np.sin(2 * np.pi * freq * t)
    return tone.astype(np.int16)


if __name__ == '__main__':
    # Generate PCM data
    pcm_data = generate_sine_wave(freq=440, duration=1)  # 440 Hz, 1 second

    # Convert to WAV bytes
    wav_bytes = pcm_arr_to_wav_bytes(
        pcm_data=pcm_data,
        sample_rate=44100,
        num_channels=1,
        bit_depth=16
    )

    # Save to a file to verify
    with open("/tmp/test_audio.wav", "wb") as f:
        f.write(wav_bytes)

    print("WAV file generated as 'test_audio.wav'")