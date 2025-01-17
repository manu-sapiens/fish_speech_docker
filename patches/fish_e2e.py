import base64
import ctypes
import io
import json
import os
import struct
import sys
from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator, Union

import httpx
import numpy as np
import ormsgpack
import soundfile as sf
from loguru import logger

from .schema import (
    ServeMessage,
    ServeRequest,
    ServeTextPart,
    ServeVQGANDecodeRequest,
    ServeVQGANEncodeRequest,
    ServeVQPart,
)

logger.info("=============== FISH_E2E.PY LOADED ===============")
logger.info(f"Current file: {__file__}")
logger.info(f"Current directory: {os.getcwd()}")
logger.info(f"Directory contents: {os.listdir('.')}")
logger.info("================================================")

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
        max_keepalive_connections=5,
        max_connections=10,
        keepalive_expiry=5.0,
    ),
)


class FishE2EAgent:
    def __init__(self):
        self.llm_url = os.getenv("LLM_URL", "http://localhost:8080/v1/chat")
        self.vqgan_url = os.getenv("VQGAN_URL", "http://localhost:8080")
        self.client = httpx.AsyncClient(timeout=None)

    async def get_codes(self, audio_data, sample_rate):
        logger.info(f"Converting audio data with shape: {audio_data.shape}")
        # Convert audio data to bytes
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, audio_data, sample_rate, format="WAV")
        audio_bytes.seek(0)

        # Send audio to VQGAN encoder
        payload = ServeVQGANEncodeRequest(audios=[audio_bytes.read()])
        logger.info("Sending request to VQGAN encoder...")
        resp = await self.client.post(
            f"{self.vqgan_url}/vqgan/encode",
            content=ormsgpack.packb(payload.dict()),
            headers={"content-type": "application/x-msgpack"},
        )
        resp.raise_for_status()
        codes = ormsgpack.unpackb(resp.content)["codes"]
        return codes

    async def stream(
        self,
        system_audio_data: np.ndarray | None,
        user_audio_data: np.ndarray | None,
        sample_rate: int,
        num_channels: int,
        chat_ctx: dict | None = None,
    ) -> AsyncGenerator[FishE2EEvent, None]:
        """
        Stream audio and text from the model.
        """
        # Prepare chat context
        if chat_ctx is None:
            chat_ctx = {"messages": []}

        # Prepare system audio
        if system_audio_data is not None:
            system_codes = await self.get_codes(system_audio_data, sample_rate)
            chat_ctx["messages"].append(
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": ""},
                        {"type": "vq", "codes": system_codes},
                    ],
                }
            )

        # Prepare user audio
        if user_audio_data is not None:
            user_codes = await self.get_codes(user_audio_data, sample_rate)
            yield FishE2EEvent(type=FishE2EEventType.USER_CODES, vq_codes=user_codes)
            chat_ctx["messages"].append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ""},
                        {"type": "vq", "codes": user_codes},
                    ],
                }
            )

        # Send request to LLM
        async with self.client.stream(
            "POST",
            self.llm_url,
            json=chat_ctx,
            headers={"Accept": "text/event-stream"},
            timeout=None,
        ) as resp:
            resp.raise_for_status()

            async for line in resp.aiter_lines():
                if not line or line.startswith(":"):
                    continue

                try:
                    event = json.loads(line.removeprefix("data: "))
                except json.JSONDecodeError:
                    continue

                # Process text
                if "text" in event:
                    print("TEXT:", event["text"])
                    yield FishE2EEvent(
                        type=FishE2EEventType.TEXT_SEGMENT, text=event["text"]
                    )

                # Process speech
                if "speech" in event:
                    # Decode base64 audio
                    frame_bytes = base64.b64decode(event["speech"])
                    logger.debug(f"Received audio frame size: {len(frame_bytes)}")

                    # Parse audio frame
                    frame = CustomAudioFrame(
                        frame_bytes,
                        sample_rate=sample_rate,
                        num_channels=num_channels,
                        samples_per_channel=len(frame_bytes)
                        // (num_channels * ctypes.sizeof(ctypes.c_int16)),
                    )
                    print(f"AUDIO: sample_rate={sample_rate}, num_channels={num_channels}")

                    # Convert to float32
                    frame_data = np.array(frame.data, dtype=np.float32) / 32768.0
                    frame_data = frame_data.reshape(-1, num_channels)

                    yield FishE2EEvent(type=FishE2EEventType.SPEECH_SEGMENT, frame=frame_data)

                # Process ASR result
                if "asr" in event:
                    yield FishE2EEvent(type=FishE2EEventType.ASR_RESULT, text=event["asr"])

                # Process end of text
                if event.get("done", False):
                    yield FishE2EEvent(type=FishE2EEventType.END_OF_TEXT)

                # Process end of speech
                if event.get("speech_done", False):
                    print("SPEECH DONE")
                    yield FishE2EEvent(type=FishE2EEventType.END_OF_SPEECH)


# Example usage:
async def main():
    import asyncio
    import soundfile as sf

    logger.info("Loading audio file test.wav...")
    # Load audio file
    audio_data, sample_rate = sf.read("test.wav")
    if len(audio_data.shape) == 1:
        audio_data = audio_data[:, None]

    # Create agent
    logger.info("Creating agent...")
    agent = FishE2EAgent()

    # Stream response
    logger.info("Starting stream...")
    async for event in agent.stream(
        system_audio_data=None,
        user_audio_data=audio_data,
        sample_rate=sample_rate,
        num_channels=audio_data.shape[1],
    ):
        if event.type == FishE2EEventType.TEXT_SEGMENT:
            print(f"Text: {event.text}")
        elif event.type == FishE2EEventType.SPEECH_SEGMENT:
            print(f"Speech frame shape: {event.frame.shape}")
        elif event.type == FishE2EEventType.ASR_RESULT:
            print(f"ASR: {event.text}")
        elif event.type == FishE2EEventType.END_OF_TEXT:
            print("End of text")
        elif event.type == FishE2EEventType.END_OF_SPEECH:
            print("End of speech")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
