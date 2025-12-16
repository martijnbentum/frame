import copy
import numpy as np

class Frames:
    '''Frames class to handle the frames of the wav2vec2 outputs
    '''
    def __init__(self,n_frames, stride = 0.02, field = 0.025,
        start_time = 0, frames = None, identifier = '', name = '',
        audio_filename = '', outputs = None):
        '''Handles the frames of the wav2vec2 outputs
        n_frames        number of frames
        stride          the time between frames
        field           the length of the frame
        start_time      the start time of the first frame
        frames          list of frames
        identifier      identifier of the outputs
        name            name of the outputs
        audio_filename  audio filename
        outputs         the outputs of the wav2vec2 model
        '''

        self.identifier = identifier
        self.name = name
        self.audio_filename = audio_filename
        self.outputs = outputs
        self.stride = stride
        self.field = field
        self.n_frames = n_frames
        self.start_time = start_time
        self._make_frames()
        self.end_time = self.frames[-1].end_time
        self.duration = self.end_time - self.frames[0].start_time
        self._set_transformer_info()

    def __repr__(self):
        m = f'Frames(#frames:{self.n_frames}, start:{self.start_time:.3f}'
        m += f', duration:{self.duration:.3f}'
        m += f', stride:{self.stride:.3f}, field:{self.field})'
        return m

    def _make_frames(self):
        self.frames = []
        for index in range(self.n_frames):
            self.frames.append(Frame(index, self.stride, self.field,
                self.start_time, self))

    def _set_transformer_info(self):
        self.transformer_none_indices = []
        self.transformer_available_indices = []
        if self.outputs is None: return
        if not hasattr(self.outputs,'hidden_states'): return
        for i, hidden_state in enumerate(self.outputs.hidden_states):
            if hidden_state is None:
                self.transformer_none_indices.append(i)
            else:
                self.transformer_available_indices.append(i)

    def start_frame(self, start_time = None, end_time = None, 
        percentage_overlap = None):
        '''Get the start frame of the frames that overlap with the start
        and end times
        '''
        selected_frames = self.select_frames(start_time, end_time,
            percentage_overlap = percentage_overlap)
        return select_start_frame(selected_frames)

    def middle_frame(self, start_time = None, end_time = None, 
        percentage_overlap = None):
        '''Get the middle frame of the frames that overlap with the start
        and end times
        '''
        selected_frames = self.select_frames(start_time, end_time,
            percentage_overlap = percentage_overlap)
        return select_middle_frame(selected_frames)

    def end_frame(self, start_time = None, end_time = None, 
        percentage_overlap = None):
        '''Get the end frame of the frames that overlap with the start
        and end times
        '''
        selected_frames = self.select_frames(start_time, end_time,
            percentage_overlap = percentage_overlap)
        return select_end_frame(selected_frames)

    def start_middle_end_frames(self, start_time = None, end_time = None,
        percentage_overlap = None):
        '''Get the start, middle and end frames of the frames that overlap
        with the start and end times
        '''
        selected_frames = self.select_frames(start_time, end_time,
            percentage_overlap = percentage_overlap)
        return select_start_middle_end_frames(selected_frames)

    def select_frames(self,start_time = None,end_time = None, 
        percentage_overlap = None):
        '''Select frames that overlap with the start and end times
        start_time         start time in seconds
        end_time           end time in seconds
        percentage_overlap the percentage of the frame that must overlap
        '''
        if start_time == end_time == None:
            return self.frames
        if start_time is None: start_time = self.start_time
        if end_time is None: end = self.end_time
        selected_frames = []
        for frame in self.frames:
            if percentage_overlap == None:
                if frame.overlap(start_time,end_time):
                    selected_frames.append(frame)
            elif frame.overlap_percentage(start,end) >= percentage_overlap:
                selected_frames.append(frame)
        return selected_frames

    def cnn(self, start_time = None, end_time = None, average = False,
        percentage_overlap = None, position = None):
        '''Get the cnn output of the frames that overlap with the start
        and end times
        '''
        if self.outputs is None or not hasattr(self.outputs,'extract_features'): 
            print('No outputs / extract_features available')
            return None
        if position not in [None, 'start', 'middle', 'end']:
            raise ValueError('position must be None, start, middle or end')
        if position:
            method = getattr(self, f'{position}_frame')
            frame = method(start_time, end_time,
                percentage_overlap = percentage_overlap)
            return frame.cnn()
        frames = self.select_frames(start_time, end_time,
            percentage_overlap = percentage_overlap)
        if len(frames) == 1: return frames[0].cnn()
        output = np.array([frame.cnn() for frame in frames])
        if average:
            return np.mean(output,axis=0)
        return output

    def feature_vector(self, start_time = None, end_time = None,
        average = False, percentage_overlap = None):
        '''Get the cnn output of the frames that overlap with the start
        and end times
        '''
        return self.cnn(start_time, end_time, average,
            percentage_overlap = percentage_overlap)

    def transformer(self, layer, start_time = None, end_time = None,
        average = False,percentage_overlap = None, position= None):
        '''Get the transformer output of the frames that overlap with the start
        and end times
        '''
        if self.outputs is None or not hasattr(self.outputs,'hidden_states'): 
            print('No outputs / hidden_states available')
            return None
        if layer not in self.transformer_available_indices:
            m = f'Layer {layer} not available in the transformer outputs\n'
            m += f'Available layers: {self.transformer_available_indices}'
            raise ValueError('Layer not available in the transformer outputs')
        if position not in [None, 'start', 'middle', 'end']:
            raise ValueError('position must be None, start, middle or end')
        if position:
            method = getattr(self, f'{position}_frame')
            frame = method(start_time, end_time,
                percentage_overlap = percentage_overlap)
            return frame.transformer(layer)
        frames = self.select_frames(start_time, end_time,
            percentage_overlap = percentage_overlap)
        if len(frames) == 1: return frames[0].transformer(layer)
        output = np.array([frame.transformer(layer) for frame in frames])
        if average:return np.mean(output,axis=0)
        return output



class Frame:
    '''Frame class to handle a single frame of the wav2vec2 output.'''
    def __init__(self, index, stride, field, global_start_time, parent,
        info = {}):
        '''Handles a single frame of the wav2vec2 outputs
        index               index of the frame
        stride              the time between frames
        field               the length of the frame
        global_start_time   the start time of the first frame
        parent              the parent Frames object
        '''
        self.index = index
        self.stride = stride
        self.field = field
        self.global_start_time = global_start_time
        self.parent = parent
        self.info = info

        self.start_time = self.index * self.stride + self.global_start_time
        self.end_time = self.start_time + self.field

    def __repr__(self):
        return f'Frame({self.index}, {self.start_time:.3f}, {self.end_time:.3f})'

    def overlap(self,start_time, end_time):
        return self.start_time < end_time and self.end_time > start_time

    def overlap_time(self,start_time, end_time):
        '''calculate the overlap time in seconds 
        between the frame and the given time range
        '''
        return min(self.end_time,end_time) - max(self.start_time,start_time)

    def overlap_percentage(self,start_time,end_time):
        '''calculate the overlap percentage
        between the frame and the given time range
        if the frame is completely within the time range, return 100%
        '''
        if self.overlap(start_time, end_time):
            ot = self.overlap_time(start_time,end_time)
            return round(ot / self.field,5) * 100
        return 0.0

    def cnn(self):
        if not hasattr(self.parent,'outputs'):return None
        if not hasattr(self.parent.outputs,'extract_features'):return None
        return self.parent.outputs.extract_features[0,self.index,:]

    def feature_vector(self):
        return self.cnn()

    def transformer(self, layer):
        if not hasattr(self.parent,'outputs'):return None
        if not hasattr(self.parent.outputs,'hidden_states'):return None
        return self.parent.outputs.hidden_states[layer][0,self.index,:]

    def to_json(self):
        d = {}
        d['index'] = self.index
        d['start_time'] = self.start_time 
        d['end_time'] = self.end_time
        d['identifier'] = self.parent.identifier
        d['info'] = self.info
        return d

def find_frame_start_time(start_time, stride = 0.02):
    '''find the start time of the first frame.'''
    if abs(0 - start_time) < 0.001: return 0
    print('using start time: {start_time}, stride {stride}')
    end_time = start_time + 0.001
    nframes = int(start_time / stride) + 5
    frames = Frames(nframes, start_time =  0)
    for i, f in enumerate(frames.frames):
        if f.overlap(start_time, end_time):
            return f.start_time
    raise ValueError('Could not find frame start time', start_time)

def make_frames_from_outputs(outputs, **kwargs):
    '''make frames object from the outputs
    outputs         the outputs of the wav2vec2 model
    '''
    output_n_frames = outputs.extract_features.shape[1]
    n_frames = output_n_frames
    frames = Frames(n_frames, **kwargs)
    return frames

def make_frames_from_duration(duration, stride = 0.02, field = 0.025,
    identifier = ''):
    nframes = int(duration / stride) - 1
    ms_duration = int(round(duration *1000))
    ms_leftover = ms_duration - int(round(nframes * stride * 1000))
    ms_field = int(round(field*1000))
    if ms_leftover >= ms_field: nframes += 1
    return Frames(nframes, stride, field, identifier = identifier)


def extract_outputs_times(outputs, start_time, end_time):
    '''extract the outputs that overlap with the start and end times
    '''
    frames = make_frames_with_outputs(outputs, start_time = start_time)
    selected_frames = frames.select_frames(start_time, end_time)
    start_index = selected_frames[0].index
    end_index = selected_frames[-1].index + 1
    start_time = selected_frames[0].start_time
    return extract_outputs_indices(outputs, start_index, end_index, start_time)

def extract_outputs_indices(outputs, start_index, end_index, start_time):
    '''extract the outputs that overlap with the start and end indices
    '''
    o = copy.deepcopy(outputs)
    o.audio_filename = outputs.audio_filename
    o.start_time = start_time
    o.identifier = outputs.identifier
    o.name = outputs.name
    o.extract_features = outputs.extract_features[:,start_index:end_index,:]
    o.hidden_states = extract_hidden_states(outputs.hidden_states,
        start_index, end_index)
    return o

def extract_hidden_states(hidden_states, start_index, end_index):
    '''extract the hidden states that overlap with the start and end indices
    '''
    hs = []
    for hidden_state in hidden_states:
        hs.append(hidden_state[:,start_index:end_index,:])
    return hs

def select_start_frame(frames):
    n_frames = len(frames)
    if n_frames > 0: return frames[0]

def select_middle_frame(frames):
    n_frames = len(frames)
    if n_frames == 1: return frames[0]
    if n_frames % 2 == 0:
        return frames[int(n_frames / 2) - 1]
    return frames[int(n_frames / 2)]

def select_end_frame(frames):
    n_frames = len(frames)
    if n_frames > 0: return frames[-1]

def select_start_middle_end_frames(frames):
    n_frames = len(frames)
    d = {'start': None, 'middle': None, 'end': None}
    if n_frames == 0: return d
    if n_frames > 0:
        d['start'] = select_start_frame(frames)
    if n_frames > 1:
        d['end'] = select_end_frame(frames)
    if n_frames == 3:
        d['middle'] = frames[1]
    elif n_frames > 3:
        d['middle'] = select_middle_frame(frames)
    return d
