# Examples

Put your video at `examples/interview.mp4`, then run the scripts from the repository root.

## ask_interview.py: ask an LLM about the video

```python
import saccade

llm = saccade.azure(
    endpoint="https://<your-resource>.openai.azure.com", api_key="<your-key>", deployment="gpt-4o"
)

video = saccade.Video("examples/interview.mp4", llm=llm)

print(video.ask("How was the interview?", progress=print))
```

```bash
pip install -e .
python examples/ask_interview.py
```

What `ask()` does:

1. Transcribes the audio and picks representative frames (the images) **on your machine**.
   This happens only the first time; the result is cached, so later questions start
   immediately.
2. Sends your LLM the timestamped transcript plus up to 12 frames from the video.
3. Tells the LLM to answer only from that evidence and to cite timestamps.

`answer.evidence` and `answer.frames` show exactly what was sent.

You can also leave the arguments out of `saccade.azure()` and set `AZURE_AI_ENDPOINT`,
`AZURE_AI_API_KEY` and `AZURE_AI_DEPLOYMENT` instead. The endpoint can be any URL the
Foundry portal shows for your deployment.

Other LLMs:

- **OpenAI:** `saccade.openai("gpt-4o")`
- **Local Ollama:** `saccade.ollama("llama3.1")`, which keeps everything offline
- **Any OpenAI-compatible server:** `saccade.OpenAICompatible(base_url=..., model=...)`

## quickstart.py: no LLM, 100% local

Shows search over what was said, the frames of what was shown, and prompt-ready context
you can paste into any LLM.
