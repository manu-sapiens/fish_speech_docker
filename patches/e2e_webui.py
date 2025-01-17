import io
import re
import wave

import gradio as gr
import numpy as np

from .fish_e2e import FishE2EAgent, FishE2EEventType
from .schema import ServeMessage, ServeTextPart, ServeVQPart


def wav_chunk_header(sample_rate=44100, bit_depth=16, channels=1):
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)

    wav_header_bytes = buffer.getvalue()
    buffer.close()
    return wav_header_bytes


class ChatState:
    def __init__(self):
        self.conversation = []
        self.added_systext = False
        self.added_sysaudio = False

    def get_history(self):
        results = []
        for msg in self.conversation:
            results.append({"role": msg.role, "content": self.repr_message(msg)})

        # Process assistant messages to extract questions and update user messages
        for i, msg in enumerate(results):
            if msg["role"] == "assistant":
                match = re.search(r"Question: (.*?)\n\nResponse:", msg["content"])
                if match and i > 0 and results[i - 1]["role"] == "user":
                    # Update previous user message with extracted question
                    results[i - 1]["content"] += "\n" + match.group(1)
                    # Remove the Question/Answer format from assistant message
                    msg["content"] = msg["content"].split("\n\nResponse: ", 1)[1]
        return results

    def repr_message(self, msg: ServeMessage):
        response = ""
        for part in msg.parts:
            if isinstance(part, ServeTextPart):
                response += part.text
            elif isinstance(part, ServeVQPart):
                response += "[Audio]"
        return response


def clear_fn():
    return ChatState()


async def process_audio_input(
    sys_audio_input, sys_text_input, audio_input, state: ChatState, text_input: str
):
    # Initialize agent
    agent = FishE2EAgent()

    # Process system message
    if not state.added_systext and sys_text_input:
        print("System text input:", sys_text_input)
        state.conversation.append(
            ServeMessage(role="system", parts=[ServeTextPart(text=sys_text_input)])
        )
        state.added_systext = True

    if not state.added_sysaudio and sys_audio_input is not None:
        if isinstance(sys_audio_input, tuple):
            sys_audio_input = sys_audio_input[1]
        state.conversation.append(
            ServeMessage(
                role="system",
                parts=[ServeTextPart(text=""), ServeVQPart(codes=[[]])],
            )
        )
        state.added_sysaudio = True

    # Process user message
    if text_input:
        print("User text input:", text_input)
        state.conversation.append(
            ServeMessage(role="user", parts=[ServeTextPart(text=text_input)])
        )

    if audio_input is not None:
        print("User audio input")
        if isinstance(audio_input, tuple):
            audio_input = audio_input[1]
        state.conversation.append(
            ServeMessage(role="user", parts=[ServeTextPart(text=""), ServeVQPart(codes=[[]])])
        )

    # Stream response
    response_text = ""
    audio_chunks = []
    wav_header = wav_chunk_header(sample_rate=24000, channels=1)

    async for event in agent.stream(
        system_audio_data=sys_audio_input,
        user_audio_data=audio_input,
        sample_rate=24000,
        num_channels=1,
        chat_ctx={"messages": [msg.dict() for msg in state.conversation]},
    ):
        if event.type == FishE2EEventType.TEXT_SEGMENT:
            response_text += event.text
            yield state.get_history(), None
        elif event.type == FishE2EEventType.SPEECH_SEGMENT:
            audio_chunk = (event.frame * 32768.0).astype(np.int16).tobytes()
            audio_chunks.append(audio_chunk)
            yield state.get_history(), (24000, wav_header + b"".join(audio_chunks))

        print("event text is", event.text)

    # Add response to conversation
    state.conversation.append(
        ServeMessage(
            role="assistant",
            parts=[ServeTextPart(text=response_text), ServeVQPart(codes=[[]])],
        )
    )
    yield state.get_history(), (24000, wav_header + b"".join(audio_chunks))


async def process_text_input(
    sys_audio_input, sys_text_input, state: ChatState, text_input: str
):
    return await process_audio_input(sys_audio_input, sys_text_input, None, state, text_input)


def create_demo():
    with gr.Blocks() as demo:
        state = gr.State(ChatState)

        with gr.Row():
            with gr.Column(scale=1):
                sys_text = gr.Textbox(
                    show_label=False,
                    placeholder="Enter system message...",
                    lines=3,
                )
                sys_audio = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="System Audio",
                )

            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    bubble_full_width=False,
                    avatar_images=(None, "🤖"),
                    height=600,
                )

                with gr.Row():
                    audio = gr.Audio(
                        sources=["microphone", "upload"],
                        type="numpy",
                        label="Input Audio",
                    )

                with gr.Row():
                    text = gr.Textbox(
                        show_label=False,
                        placeholder="Enter text and press enter, or upload an audio file",
                        lines=3,
                    )

                with gr.Row():
                    submit = gr.Button("Submit")
                    clear = gr.Button("Clear")

        # Event handlers
        submit_event = submit.click(
            process_audio_input,
            [sys_audio, sys_text, audio, state, text],
            [chatbot, audio],
        )

        text_event = text.submit(
            process_text_input,
            [sys_audio, sys_text, state, text],
            [chatbot, audio],
        )

        clear_event = clear.click(clear_fn, [], [state])
        clear_event.then(lambda: [], [], [chatbot])
        clear_event.then(lambda: None, [], [audio])
        clear_event.then(lambda: "", [], [text])
        clear_event.then(lambda: None, [], [sys_audio])
        clear_event.then(lambda: "", [], [sys_text])

        submit_event.then(lambda: None, [], [audio])
        submit_event.then(lambda: "", [], [text])

        text_event.then(lambda: "", [], [text])

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, root_path="/")
