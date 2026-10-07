import saccade

# Reads AZURE_AI_ENDPOINT, AZURE_AI_API_KEY and AZURE_AI_DEPLOYMENT from the environment.
video = saccade.Video("examples/interview.mp4", llm=saccade.azure())

answer = video.ask("What is the color of Gabriel's T-shirt?", progress=print)

print(answer)
print(f"Tokens: {answer.input_tokens} in + {answer.output_tokens} out = {answer.total_tokens}")
