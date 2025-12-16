# w2v-frames

Utilities for working with frame-level representations from wav2vec2
outputs.

## Installation

```bash
pip install git+ssh://git@github.com/you/w2v-frames.git

```python
from w2v_frames import make_frames_from_outputs

frames = make_frames_from_outputs(outputs) # is the output from wav2vec2 model from the transformers library
middle = frames.middle_frame(start = .3, end = .5) # get middle frame between 300ms and 500ms
print(middle)
```

