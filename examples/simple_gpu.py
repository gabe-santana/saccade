import saccade

# Same as simple_cpu.py, but Whisper runs on an NVIDIA GPU (pip install -e ".[gpu]")
# and frames are taken from keyframes only to reduce frame extraction work.
video = saccade.Video(
    "examples/video.mp4",
    llm=saccade.azure(reasoning_effort="low"),  # fewer hidden "thinking" tokens
    device="cuda",
    visual=saccade.VisualConfig(keyframes_only=True),
)

answer = video.ask("Summarize the content of the video", progress=print)

print(answer)
print(
    f"Tokens: {answer.input_tokens} in ({answer.cached_tokens} cached, billed at a discount)"
    f" + {answer.output_tokens} out = {answer.total_tokens}"
)
