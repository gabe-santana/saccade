"""Saccade without an LLM: everything runs locally and nothing leaves your machine."""

import saccade

video = saccade.Video("examples/interview.mp4")
video.index(progress=print)

for hit in video.search("experience"):  # what was said
    print(hit)

for frame in video.frames():  # what was shown
    print(frame)

print(video.context("What experience does the candidate have?").text)  # paste into any LLM
