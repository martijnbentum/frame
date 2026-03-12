from . import audio
from . import frames 

class Data:
    def __init__(self, labels, target_index, collar = 1.0, 
        min_target_duration = .075, audio_duration = None, audio_info = None,
        audio_filename = None):
        self.labels = labels
        self.label_strs = [label.label for label in labels]
        self.target_index = target_index
        self.collar = collar
        self.min_target_duration = min_target_duration
        self._handle_audio(audio_duration, audio_info, audio_filename)
        self.target = labels[target_index]
        self.target_duration=self.target.end_seconds-self.target.start_seconds
        self.start_context = labels_to_start(labels)
        self.target_mid_point = middle_point(self.target.start_seconds, 
            self.target.end_seconds)
        self.end_context = labels_to_end(labels)
        self.start = self.start_context - collar
        if self.start < 0: self.start = 0
        self.end = self.end_context + collar
        self.duration = self.end - self.start
        if self.audio_duration is not None and self.end > self.audio_duration:
            self.end = self.audio_duration
        self.frames = labels_to_frames(labels, self.start, self.end)
        self.target_frames = self.target.frames
        self.target_center_frame = self.target.center_frame
        self.check_ok()
        if self.ok: 
            self._set_preceding_and_following_labels()
            self._set_preceding_and_following_frames_without_label()
            self._define_neighbours()

    def __repr__(self):
        frame = f'#frames:{self.frames.n_frames}, ' 
        frame += f'start:{self.frames.start_time:.3f}, '
        frame += f'end:{self.frames.end_time:.3f}'
        return f'Data(target={self.target}, {frame}, ok={self.ok}'

    def __str__(self):
        m = self.__repr__() + '\n'
        m += f'preceding frames:\n'
        m += f'  first frame: {self._first_or_none(self.preceding_frames_no_label)}\n'
        m += f'  #frames without label: {len(self.preceding_frames_no_label)}\n'
        m += f'  last frame: {self._last_or_none(self.preceding_frames_no_label)}\n' 
        m += f'preceding labels:\n'
        m += f'  first frame: {self._label_boundary_frame(self.preceding_labels, first = True)}\n'
        for label in self.preceding_labels:
            m += f'  {label}, center: {label.center_frame}'
            m += f' #frames: {len(label.frames)}\n'
        m += f'  last frame: {self._label_boundary_frame(self.preceding_labels, first = False)}\n'
        m += f'target label: \n'
        m += f'  first frame: {self.target.frames[0]}\n'
        m += f'  {self.target}, center: {self.target.center_frame}'
        m += f' #frames: {len(self.target.frames)}\n'
        m += f'  last frame: {self.target.frames[-1]}\n'
        m += f'following labels: \n'
        m += f'  first frame: {self._label_boundary_frame(self.following_labels, first = True)}\n'
        for label in self.following_labels:
            m += f'  {label}, center: {label.center_frame}'
            m += f' #frames: {len(label.frames)}\n'
        m += f'  last frame: {self._label_boundary_frame(self.following_labels, first = False)}\n'
        m += f'following frames:\n'
        m += f'  first frame: {self._first_or_none(self.following_frames_no_label)}\n'
        m += f'  #frames without label: {len(self.following_frames_no_label)}\n'
        m += f'  last frame: {self._last_or_none(self.following_frames_no_label)}\n'
        return m

    def _first_or_none(self, items):
        if not items:
            return None
        return items[0]

    def _last_or_none(self, items):
        if not items:
            return None
        return items[-1]

    def _label_boundary_frame(self, labels, first = True):
        if not labels:
            return None
        frames = labels[0].frames if first else labels[-1].frames
        if not frames:
            return None
        return frames[0] if first else frames[-1]

    def _set_preceding_and_following_labels(self):
        if self.target_index <= 0: self.preceding_labels = []
        else:self.preceding_labels = self.labels[:self.target_index]
        if self.target_index >= len(self.labels) - 1: self.following_labels = []
        else:self.following_labels = self.labels[self.target_index + 1:]

    def _set_preceding_and_following_frames_without_label(self):
        self.preceding_frames_no_label = []
        self.following_frames_no_label = []
        for frame in self.frames.frames:
            if frame.label is None:
                if frame.start_time <= self.target.start_seconds:
                    self.preceding_frames_no_label.append(frame)
                elif frame.start_time >= self.target.end_seconds:
                    self.following_frames_no_label.append(frame)

    def _define_neighbours(self):
        n = min(len(self.preceding_labels), len(self.following_labels))
        order_names = 'first', 'second', 'third', 'fourth', 'fifth', 'sixth'
        for on, i in zip(order_names, range(n)):
            prev = self.preceding_labels[::-1][i]
            follow = self.following_labels[i]
            frames = prev.frames + follow.frames
            center_frames = prev.center_frame, follow.center_frame
            setattr(self, f'{on}_neighbour', (prev, follow))
            setattr(self, f'{on}_neighbour_frames', frames)
            setattr(self, f'{on}_neighbour_center_frames', center_frames)
        
        

    def check_ok(self):
        self.ok = True
        if self.audio_duration is None: self.ok = None
        elif self.audio_duration < self.end_context: self.ok = False
        if self.target_duration < self.min_target_duration: self.ok = False

    def _handle_audio(self, audio_duration, audio_info, audio_filename):
        self.audio_filename = audio_filename
        self.audio_duration = audio_duration
        self.audio_info = audio_info
        if audio_filename: self.audio_info = audio.audio_info(audio_filename)
        if self.audio_info:
            self.audio_duration = self.audio_info['duration']
            self.audio_filename = self.audio_info['filename']

    

def labels_to_frames(labels, start = None, end = None, stride = 0.02, 
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
    if start is None: start = labels_to_start(labels)
    if end is None: end = labels_to_end(labels)
    duration = end - start
    temp = frames.make_frames_from_duration(duration, stride, field)
    f = frames.Frames(temp.n_frames, start_time = start)
    for label in labels:
        label.frames = []
        label.all_frames = []
        set_center_frame_to_label(label, f.frames)
    for frame in f.frames:
        frame.labels = []
        frame.label = None
        for label in labels:
            start, end = label.start_seconds, label.end_seconds
            if frame.overlap_percentage(start, end) > .1:
                frame.labels.append(label)
                label.all_frames.append(frame)
        if len(frame.labels) > 2: 
            m  = f'{frame} overlaps with more than 2 labels: {frame.labels}'
            raise ValueError(m)
        if len(frame.labels) == 0: 
            frame.ok = False 
            continue
        if len(frame.labels) == 1: 
            frame.label = frame.labels[0]
            frame.label.frames.append(frame)
            continue
        frame.label = find_label_with_max_overlap(frame, frame.labels)
        frame.label.frames.append(frame)
    return f

def middle_point(start, end):
    return (start + end) / 2

def labels_to_end(labels):
    end = max(label.end_seconds for label in labels)
    return end
    
def labels_to_start(labels):
    start = min(label.start_seconds for label in labels)
    return start
    
def set_center_frame_to_label(label, frames):
    label.mid_point = middle_point(label.start_seconds, label.end_seconds)
    label.center_frame = None
    start, end = label.mid_point - 0.01, label.mid_point + 0.01
    label.center_frame = find_frame_with_max_overlap(start, end,frames)
        

def find_frame_with_max_overlap(start,end, overlapping_frames):
    max_perc = 0
    selected_frame = None
    for frame in overlapping_frames:
        o = frame.overlap_percentage(start, end)
        if o > max_perc:
            max_perc = o
            selected_frame = frame
    # print(f'max perc {max_perc}, selected_frame {selected_frame}')
    return selected_frame
        
def find_label_with_max_overlap(frame, overlapping_labels):
    max_perc = 0
    selected_label = None

    for label in overlapping_labels:
        o = frame.overlap_percentage(label.start_seconds, label.end_seconds)
        if o == 0: continue
        if o > max_perc:
            max_perc = o
            selected_label = label
        if o == max_perc:
            d1 = label.end_seconds - label.start_seconds
            d2 = selected_label.end_seconds - selected_label.start_seconds
            if d1 < d2: selected_label = label
    return selected_label

