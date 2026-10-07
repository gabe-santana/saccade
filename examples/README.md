# Examples: ask an LLM about a video

| Script | Runs Whisper on | Frame sampling |
|---|---|---|
| [simple_cpu.py](simple_cpu.py) | CPU | Default sampling |
| [simple_gpu.py](simple_gpu.py) | NVIDIA GPU | Keyframes only |

Indexing time depends on your hardware, video length, and processing settings.
The index is cached and reused on subsequent runs with the same video and settings.

Both scripts ask an Azure AI Foundry model to summarize a video, then print the answer
and token usage. Change the question to ask about your own video's content:

```python
import saccade

# Reads AZURE_AI_ENDPOINT, AZURE_AI_API_KEY and AZURE_AI_DEPLOYMENT from the environment.
video = saccade.Video("examples/video.mp4", llm=saccade.azure())

answer = video.ask("Summarize the content of the video.", progress=print)

print(answer)
print(f"Tokens: {answer.input_tokens} in + {answer.output_tokens} out = {answer.total_tokens}")
```

The GPU version adds two options to `saccade.Video`:

- `device="cuda"` runs Whisper on the GPU.
- `visual=saccade.VisualConfig(keyframes_only=True)` takes frames from the video's keyframes
  only, so frame extraction keeps up with the GPU.

## Run it

1. Install Saccade from the repository root. For the GPU example, include the `gpu` extra:

   ```bash
   pip install -e .            # CPU
   pip install -e ".[gpu]"     # GPU: adds NVIDIA's CUDA libraries
   ```

   The GPU needs an NVIDIA card with a recent driver. The CUDA libraries come from pip, so
   there is no separate CUDA toolkit to install.

2. Put your video at `examples/video.mp4`, or change the path in the script to your video.
   Video files are git-ignored, so they won't be committed.

3. Set your Azure AI Foundry connection. The values come from the Foundry portal, under your
   deployment:

   ```bash
   export AZURE_AI_ENDPOINT="https://<your-resource>.services.ai.azure.com/openai/v1/responses"
   export AZURE_AI_API_KEY="<your-key>"
   export AZURE_AI_DEPLOYMENT="<your-deployment-name>"   # e.g. gpt-4o or gpt-5-mini
   ```

   In PowerShell, set the same variables like this instead:

   ```powershell
   $env:AZURE_AI_ENDPOINT = "..."
   ```

4. Run one of the examples from the repository root:

   ```bash
   python examples/simple_cpu.py
   python examples/simple_gpu.py
   ```

The first run processes the video on your machine: it transcribes the audio and picks
representative frames, printing progress as it goes. The result is cached, so later runs,
and any other question you ask, go straight to the LLM.

The CPU and GPU versions keep separate caches, so the first GPU run processes the video once
more.

Only the transcript text and up to 12 frames are sent to your Azure deployment, never the
video file. Keep your key in environment variables and out of the code, so it never gets
committed.
