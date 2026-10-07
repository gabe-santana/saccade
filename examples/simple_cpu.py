import saccade

# Reads AZURE_AI_ENDPOINT, AZURE_AI_API_KEY and AZURE_AI_DEPLOYMENT from the environment.
# reasoning_effort="low" makes reasoning models (GPT-5) spend fewer hidden "thinking" tokens.
video = saccade.Video("examples/video.mp4", llm=saccade.azure(reasoning_effort="low"))

answer = video.ask("Summarize the content of the video.", progress=print)

print(answer)
print(
    f"Tokens: {answer.input_tokens} in ({answer.cached_tokens} cached, billed at a discount)"
    f" + {answer.output_tokens} out = {answer.total_tokens}"
)
