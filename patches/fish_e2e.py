import base64
import ctypes
import io
import json
import os
import struct
from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator, Union

import httpx
import numpy as np
import ormsgpack
import soundfile as sf

from tools.schema import (
    ServeMessage,
    ServeTextPart,
    ServeVQGANDecodeRequest,
    ServeVQGANEncodeRequest,
    ServeVQPart,
)

from tools.schema import ServeRequest as ServeChatRequest


class CustomAudioFrame:
    def __init__(self, data, sample_rate, num_channels, samples_per_channel):
        if len(data) < num_channels * samples_per_channel * ctypes.sizeof(
            ctypes.c_int16
        ):
            raise ValueError(
                "data length must be >= num_channels * samples_per_channel * sizeof(int16)"
            )

        self._data = bytearray(data)
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._samples_per_channel = samples_per_channel

    @property
    def data(self):
        return memoryview(self._data).cast("h")

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def num_channels(self):
        return self._num_channels

    @property
    def samples_per_channel(self):
        return self._samples_per_channel

    @property
    def duration(self):
        return self.samples_per_channel / self.sample_rate

    def __repr__(self):
        return (
            f"CustomAudioFrame(sample_rate={self.sample_rate}, "
            f"num_channels={self.num_channels}, "
            f"samples_per_channel={self.samples_per_channel}, "
            f"duration={self.duration:.3f})"
        )


class FishE2EEventType(Enum):
    SPEECH_SEGMENT = 1
    TEXT_SEGMENT = 2
    END_OF_TEXT = 3
    END_OF_SPEECH = 4
    ASR_RESULT = 5
    USER_CODES = 6


@dataclass
class FishE2EEvent:
    type: FishE2EEventType
    frame: np.ndarray = None
    text: str = None
    vq_codes: list[list[int]] = None


client = httpx.AsyncClient(
    timeout=None,
    limits=httpx.Limits(
        max_connections=None,
        max_keepalive_connections=None,
        keepalive_expiry=None,
    ),
)


class FishE2EAgent:
    def __init__(self):
        self.llm_url = os.getenv("LLM_URL", "http://fish-agent:8080/v1/chat")
        self.vqgan_url = os.getenv("VQGAN_URL", "http://fish-agent:8080")
        self.client = httpx.AsyncClient(timeout=None)

    async def get_codes(self, audio_data, sample_rate):
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_data, sample_rate, format="WAV")
        audio_buffer.seek(0)
        # Step 1: Encode audio using VQGAN
        encode_request = ServeVQGANEncodeRequest(audios=[audio_buffer.read()])
        encode_request_bytes = ormsgpack.packb(
            encode_request, option=ormsgpack.OPT_SERIALIZE_PYDANTIC
        )
        encode_response = await self.client.post(
            f"{self.vqgan_url}/v1/vqgan/encode",
            data=encode_request_bytes,
            headers={"Content-Type": "application/msgpack"},
        )
        encode_response_data = ormsgpack.unpackb(encode_response.content)
        codes = encode_response_data["tokens"][0]
        return codes

    async def get_asr(self, audio_data, sr, client=None):
        """Get ASR result for audio data."""
        try:
            if client is None:
                client = httpx.AsyncClient()

            # Convert audio to WAV format
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio_data, sr, format="WAV")
            audio_buffer.seek(0)

            # Send to ASR endpoint - use the LLM URL since ASR is part of the same API server
            base_url = self.llm_url.rstrip("/")  # Remove any trailing slash
            if "/v1/chat" in base_url:  # Remove /v1/chat if present
                base_url = base_url.replace("/v1/chat", "")

            asr_response = await client.post(
                f"{base_url}/v1/asr",
                data=ormsgpack.packb(
                    {
                        "audios": [audio_buffer.read()],
                        "sample_rate": sr,
                        "language": "auto"
                    },
                    option=ormsgpack.OPT_SERIALIZE_PYDANTIC
                ),
                headers={"Content-Type": "application/msgpack"},
            )
            
            if asr_response.status_code != 200:
                print(f"ASR request failed with status {asr_response.status_code}")
                return "<ASR endpoint error>"

            asr_result = ormsgpack.unpackb(asr_response.content)
            if not isinstance(asr_result, dict):
                print(f"Unexpected ASR response format: {type(asr_result)}")
                return "<Invalid ASR response>"

            transcriptions = asr_result.get("transcriptions", [])
            if isinstance(transcriptions, list) and transcriptions:
                return transcriptions[0]
            else:
                print(f"No transcriptions in ASR response: {asr_result}")
                return "<No ASR text>"

        except Exception as e:
            print(f"ASR error: {str(e)}")
            return "<ASR failed>"

    async def stream(
        self,
        sys_audio_data,
        audio_data,
        sr,
        num_channels,
        chat_ctx=None,
        debug_mode=False,
        debug_llm=False,
        debug_asr=False,
    ):
        """Stream audio events."""
        if debug_mode:
            print("Debug mode enabled")
            if debug_llm:
                print("Debug LLM mode enabled - will process through LLM")
            else:
                print("Debug LLM mode disabled - will only test encode/decode")
            if debug_asr:
                print("Debug ASR mode enabled - will test ASR on input/output audio")

            if audio_data is not None:
                # Save input audio for comparison
                os.makedirs("outputs", exist_ok=True)
                input_path = os.path.join("outputs", "debug_input_audio.wav")
                sf.write(input_path, audio_data, sr)
                print(f"Saved input audio to {input_path}")
                print(f"Input audio shape: {audio_data.shape}, dtype: {audio_data.dtype}, range: [{audio_data.min()}, {audio_data.max()}]")

                if debug_asr:
                    # Get ASR result for input audio
                    input_text = await self.get_asr(audio_data, sr)
                    print(f"Input audio ASR result: {input_text}")

                # Convert audio to WAV format
                audio_buffer = io.BytesIO()
                sf.write(audio_buffer, audio_data, sr, format="WAV")
                audio_buffer.seek(0)

                async with httpx.AsyncClient() as client:
                    # Encode the audio
                    encode_response = await client.post(
                        f"{self.vqgan_url}/v1/vqgan/encode",
                        data=ormsgpack.packb(
                            ServeVQGANEncodeRequest(audios=[audio_buffer.read()]),
                            option=ormsgpack.OPT_SERIALIZE_PYDANTIC
                        ),
                        headers={"Content-Type": "application/msgpack"},
                    )
                    encode_response_data = ormsgpack.unpackb(encode_response.content)
                    codes = encode_response_data["tokens"][0]
                    print(f"Encode max length: {len(codes) / 21:.2f}s")

                    if not debug_llm:
                        # Direct encode/decode test without LLM
                        decode_response = await client.post(
                            f"{self.vqgan_url}/v1/vqgan/decode",
                            data=ormsgpack.packb(
                                ServeVQGANDecodeRequest(tokens=[codes]),
                                option=ormsgpack.OPT_SERIALIZE_PYDANTIC
                            ),
                            headers={"Content-Type": "application/msgpack"},
                        )
                        raw_decoded = ormsgpack.unpackb(decode_response.content)
                        
                        if isinstance(raw_decoded, dict) and 'audios' in raw_decoded:
                            # Convert bytes to audio data using numpy
                            audio_bytes = raw_decoded['audios'][0]
                            decoded_audio = np.frombuffer(audio_bytes, dtype=np.float16)
                            # Convert float16 [-1, 1] to int16 range
                            decoded_audio = (decoded_audio * 32768).astype(np.int16)
                            print(f"Decoded audio shape: {decoded_audio.shape}, dtype: {decoded_audio.dtype}, range: [{decoded_audio.min()}, {decoded_audio.max()}]")

                            if debug_asr:
                                # Get ASR result for decoded audio
                                decoded_text = await self.get_asr(decoded_audio, sr, client)
                                print(f"Decoded audio ASR result: {decoded_text}")
                        else:
                            print(f"Missing 'audios' key in response or wrong format. Keys: {list(raw_decoded.keys())}")
                            return

                        # Save decoded audio for troubleshooting
                        os.makedirs("outputs", exist_ok=True)
                        output_path = os.path.join("outputs", "debug_decoded_audio.wav")
                        sf.write(output_path, decoded_audio, sr)
                        print(f"Saved decoded audio to {output_path}")

                        yield FishE2EEvent(
                            type=FishE2EEventType.SPEECH_SEGMENT,
                            frame=CustomAudioFrame(
                                decoded_audio.tobytes(),
                                sr,
                                1,  # num_channels
                                len(decoded_audio),
                            ),
                            vq_codes=codes,
                        )
                        return

                    # If debug_llm is true, continue with LLM processing
                    yield FishE2EEvent(type=FishE2EEventType.USER_CODES, vq_codes=codes)

        if sys_audio_data is not None:
            sys_codes = await self.get_codes(sys_audio_data, sr)
        else:
            sys_codes = None
        if audio_data is not None:
            user_codes = await self.get_codes(audio_data, sr)
        # Step 2: Prepare LLM request
        if chat_ctx is None:
            sys_parts = [
                ServeTextPart(
                    text='You are a voice assistant created by Fish Audio, offering end-to-end voice interaction for a seamless user experience. You are required to first transcribe the user\'s speech, then answer it in the following format: "Question: [USER_SPEECH]\n\nAnswer: [YOUR_RESPONSE]\n". You are required to use the following voice in this conversation.'
                ),
            ]
            if sys_audio_data is not None:
                sys_parts.append(ServeVQPart(codes=sys_codes))
            chat_ctx = {
                "messages": [
                    ServeMessage(
                        role="system",
                        parts=sys_parts,
                    ),
                ],
            }
        else:
            if chat_ctx["added_sysaudio"] is False and sys_codes:
                chat_ctx["added_sysaudio"] = True
                chat_ctx["messages"][0].parts.append(ServeVQPart(codes=sys_codes))

        prev_messages = chat_ctx["messages"].copy()
        if audio_data is not None:
            yield FishE2EEvent(
                type=FishE2EEventType.USER_CODES,
                vq_codes=user_codes,
            )
        else:
            user_codes = None

        request = ServeChatRequest(
            messages=prev_messages
            + (
                [
                    ServeMessage(
                        role="user",
                        parts=[ServeVQPart(codes=user_codes)],
                    )
                ]
                if user_codes
                else []
            ),
            streaming=True,
            num_samples=1,
        )

        # Step 3: Stream LLM response and decode audio
        buffer = b""
        vq_codes = []
        current_vq = False

        async def decode_send():
            nonlocal current_vq
            nonlocal vq_codes

            data = np.concatenate(vq_codes, axis=1).tolist()
            # Decode VQ codes to audio
            decode_request = ServeVQGANDecodeRequest(tokens=[data])
            decode_response = await self.client.post(
                f"{self.vqgan_url}/v1/vqgan/decode",
                data=ormsgpack.packb(
                    decode_request,
                    option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
                ),
                headers={"Content-Type": "application/msgpack"},
            )
            decode_data = ormsgpack.unpackb(decode_response.content)

            # Convert float16 audio data to int16
            audio_data = np.frombuffer(decode_data["audios"][0], dtype=np.float16)
            audio_data = (audio_data * 32768).astype(np.int16).tobytes()

            audio_frame = CustomAudioFrame(
                data=audio_data,
                samples_per_channel=len(audio_data) // 2,
                sample_rate=44100,
                num_channels=1,
            )
            yield FishE2EEvent(
                type=FishE2EEventType.SPEECH_SEGMENT,
                frame=audio_frame,
                vq_codes=data,
            )

            current_vq = False
            vq_codes = []

        async with self.client.stream(
            "POST",
            self.llm_url,
            data=ormsgpack.packb(request, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
            headers={"Content-Type": "application/msgpack"},
        ) as response:

            async for chunk in response.aiter_bytes():
                buffer += chunk

                while len(buffer) >= 4:
                    read_length = struct.unpack("I", buffer[:4])[0]
                    if len(buffer) < 4 + read_length:
                        break

                    body = buffer[4 : 4 + read_length]
                    buffer = buffer[4 + read_length :]
                    data = ormsgpack.unpackb(body)

                    if data["delta"] and data["delta"]["part"]:
                        if current_vq and data["delta"]["part"]["type"] == "text":
                            async for event in decode_send():
                                yield event
                        if data["delta"]["part"]["type"] == "text":
                            yield FishE2EEvent(
                                type=FishE2EEventType.TEXT_SEGMENT,
                                text=data["delta"]["part"]["text"],
                            )
                        elif data["delta"]["part"]["type"] == "vq":
                            vq_codes.append(np.array(data["delta"]["part"]["codes"]))
                            current_vq = True

        if current_vq and vq_codes:
            async for event in decode_send():
                yield event

        yield FishE2EEvent(type=FishE2EEventType.END_OF_TEXT)
        yield FishE2EEvent(type=FishE2EEventType.END_OF_SPEECH)


# Example usage:
async def main():
    import torchaudio

    agent = FishE2EAgent()

    # Replace this with actual audio data loading
    with open("uz_story_en.m4a", "rb") as f:
        audio_data = f.read()

    audio_data, sample_rate = torchaudio.load("uz_story_en.m4a")
    audio_data = (audio_data.numpy() * 32768).astype(np.int16)

    stream = agent.stream(audio_data, sample_rate, 1)
    if os.path.exists("audio_segment.wav"):
        os.remove("audio_segment.wav")

    async for event in stream:
        if event.type == FishE2EEventType.SPEECH_SEGMENT:
            # Handle speech segment (e.g., play audio or save to file)
            with open("audio_segment.wav", "ab+") as f:
                f.write(event.frame.data)
        elif event.type == FishE2EEventType.ASR_RESULT:
            print(event.text, flush=True)
        elif event.type == FishE2EEventType.TEXT_SEGMENT:
            print(event.text, flush=True, end="")
        elif event.type == FishE2EEventType.END_OF_TEXT:
            print("\nEnd of text reached.")
        elif event.type == FishE2EEventType.END_OF_SPEECH:
            print("End of speech reached.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
