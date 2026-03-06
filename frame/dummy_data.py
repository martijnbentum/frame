import string

class Dummy:
    def __init__(self, start_seconds=0.0, end_seconds=0.0, label=''):
        self.start_seconds = start_seconds
        self.end_seconds = end_seconds
        self.label = label
        
    def __repr__(self):
        return f'{self.label}({self.start_seconds:.3f}-{self.end_seconds:.3f})'

def generate_segments(n=6, start=0.0, duration=0.5):
    '''generate consecutive segments.
    n               number of segments
    start           starting time in seconds
    duration        duration per segment in seconds
    '''
    segments = []
    current = start
    labels = list(string.ascii_uppercase)

    for i in range(n):
        seg = Dummy()
        seg = Dummy(current, current + duration, label=labels[i])
        segments.append(seg)
        current += duration

    return segments
