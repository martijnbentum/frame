# w2v-frames

Utilities for working with frame-level representations from wav2vec2
outputs.

## Installation

```bash
pip install git+ssh://git@github.com/you/w2v-frames.git
```

## Example Usage

```python
from w2v_frames import make_frames_from_outputs

# is the output from wav2vec2 model from the transformers library
frames = make_frames_from_outputs(outputs) 
# get middle frame between 300ms and 500ms
middle = frames.middle_frame(start = .3, end = .5) 
print(middle)
```

