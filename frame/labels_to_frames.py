import . frames 

def labels_to_frames(labels, start= None, end = None, stride = 0.02, 
    field = 0.025, time_format = 's'):

    '''
    Convert a list of labels to a list of frames.

    labels :    A list of labels, where each label is an object with attributes 
                (start_seconds, end_seconds, label).
    start :     The start time of the frames. If None, it will be set to the
                minimum start time of the labels.
    end :       The end time of the frames. If None, it will be set to the 
                maximum end time of the labels.
    stride :    The time interval between consecutive frames.
    field :     The time interval around each label to consider for frame 
                generation.

    Returns:
        list: A list of frames, where each frame is a tuple (time, label).
    '''
    if start is None: start = min(x.start_seconds for label in labels)
    if end is None: end = max(x.end_seconds for label in labels)
    duration = end - start
    f = frames.make_frames_from_duration(duration, stride, field)
    for frame in f.frames:
        frame.labels = []
        for label in labels:
            if frame.overlap(label.start_seconds, label.end_seconds):
                frame.labels.append(label.label)
    return f

