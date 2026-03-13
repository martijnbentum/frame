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
        if self.n_frames <= 0:
            raise ValueError('n_frames must be greater than 0')
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
        self.attention_none_indices = []
        self.attention_available_indices = []
        if self.outputs is None:
            return
        if (hasattr(self.outputs,'hidden_states') and
            self.outputs.hidden_states is not None):
            for i, hidden_state in enumerate(self.outputs.hidden_states):
                if hidden_state is None:
                    self.transformer_none_indices.append(i)
                else:
                    self.transformer_available_indices.append(i)
        if (hasattr(self.outputs,'attentions') and
            self.outputs.attentions is not None):
            for i, attention in enumerate(self.outputs.attentions):
                if attention is None:
                    self.attention_none_indices.append(i)
                else:
                    self.attention_available_indices.append(i)

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
        if end_time is None: end_time = self.end_time
        selected_frames = []
        po = percentage_overlap
        for frame in self.frames:
            if percentage_overlap is None:
                if frame.overlap(start_time,end_time):
                    selected_frames.append(frame)
            elif frame.overlap_percentage(start_time, end_time) >= po:
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

    def attention_query(self, layer, start_time = None, end_time = None,
        head = None, percentage_overlap = None, position = None):
        '''Get attention for selected query frames over all key frames.'''
        if self.outputs is None or not hasattr(self.outputs,'attentions'):
            print('No outputs / attentions available')
            return None
        if self.outputs.attentions is None:
            print('No outputs / attentions available')
            return None
        if layer not in self.attention_available_indices:
            m = f'Layer {layer} not available in the attention outputs\n'
            m += f'Available layers: {self.attention_available_indices}'
            raise ValueError(m)
        if position not in [None, 'start', 'middle', 'end']:
            raise ValueError('position must be None, start, middle or end')
        if position:
            method = getattr(self, f'{position}_frame')
            frame = method(start_time, end_time,
                percentage_overlap = percentage_overlap)
            return frame.attention_query(layer, head = head)
        frames = self.select_frames(start_time, end_time,
            percentage_overlap = percentage_overlap)
        if len(frames) == 1:
            return frames[0].attention_query(layer, head = head)
        return np.array([
            frame.attention_query(layer, head = head) for frame in frames
        ])

    def attention_query_key(self, layer, key_index, start_time = None,
        end_time = None, head = None, percentage_overlap = None,
        position = None):
        '''Get attention from selected query frames to a key frame.'''
        if position not in [None, 'start', 'middle', 'end']:
            raise ValueError('position must be None, start, middle or end')
        if position:
            method = getattr(self, f'{position}_frame')
            frame = method(start_time, end_time,
                percentage_overlap = percentage_overlap)
            return frame.attention_query_key(layer, key_index, head = head)
        frames = self.select_frames(start_time, end_time,
            percentage_overlap = percentage_overlap)
        if len(frames) == 1:
            return frames[0].attention_query_key(layer, key_index, head = head)
        return np.array([
            frame.attention_query_key(layer, key_index, head = head)
            for frame in frames
        ])



class Frame:
    '''Frame class to handle a single frame of the wav2vec2 output.'''
    def __init__(self, index, stride, field, global_start_time, parent,
        info = None):
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
        self.info = {} if info is None else info

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
        if self.parent.outputs.hidden_states is None:return None
        if self.parent.outputs.hidden_states[layer] is None:return None
        return self.parent.outputs.hidden_states[layer][0,self.index,:]

    def attention_query(self, layer, head = None):
        '''Get attention from this frame as query over all key frames.'''
        if not hasattr(self.parent,'outputs'):return None
        if not hasattr(self.parent.outputs,'attentions'):return None
        if self.parent.outputs.attentions is None:return None
        attention = self.parent.outputs.attentions[layer]
        if attention is None:return None
        if head is None:
            return attention[0,:,self.index,:]
        return attention[0,head,self.index,:]

    def attention_key(self, layer, head = None):
        '''Get attention to this frame as key from all query frames.'''
        if not hasattr(self.parent,'outputs'):return None
        if not hasattr(self.parent.outputs,'attentions'):return None
        if self.parent.outputs.attentions is None:return None
        attention = self.parent.outputs.attentions[layer]
        if attention is None:return None
        if head is None:
            return attention[0,:,:,self.index]
        return attention[0,head,:,self.index]

    def attention_query_key(self, layer, key_index, head = None):
        '''Get attention from this query frame to a specific key frame.'''
        query_attention = self.attention_query(layer, head = head)
        if query_attention is None:return None
        return query_attention[...,key_index]

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
    print(f'using start time: {start_time}, stride {stride}')
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
    n_frames = determine_n_frames_from_outputs(outputs)
    frames = Frames(n_frames, outputs = outputs, **kwargs)
    return frames

def make_frames_from_duration(duration, stride = 0.02, field = 0.025,
    identifier = ''):
    if duration <= 0:
        raise ValueError('duration must be greater than 0')
    nframes = int(duration / stride) - 1
    ms_duration = int(round(duration *1000))
    ms_leftover = ms_duration - int(round(nframes * stride * 1000))
    ms_field = int(round(field*1000))
    if ms_leftover >= ms_field: nframes += 1
    if nframes <= 0:
        nframes = 1
    return Frames(nframes, stride, field, identifier = identifier)


def extract_outputs_times(outputs, start_time, end_time):
    '''extract the outputs that overlap with the start and end times
    '''
    frames = make_frames_from_outputs(outputs, start_time = start_time)
    selected_frames = frames.select_frames(start_time, end_time)
    start_index = selected_frames[0].index
    end_index = selected_frames[-1].index + 1
    start_time = selected_frames[0].start_time
    return extract_outputs_indices(outputs, start_index, end_index, start_time)

def extract_outputs_indices(outputs, start_index, end_index, start_time):
    '''extract the outputs that overlap with the start and end indices
    '''
    o = copy.deepcopy(outputs)
    if hasattr(outputs, 'audio_filename'):
        o.audio_filename = outputs.audio_filename
    o.start_time = start_time
    if hasattr(outputs, 'identifier'):
        o.identifier = outputs.identifier
    if hasattr(outputs, 'name'):
        o.name = outputs.name
    if hasattr(outputs, 'extract_features') and outputs.extract_features is not None:
        o.extract_features = outputs.extract_features[:,start_index:end_index,:]
    if hasattr(outputs, 'hidden_states'):
        o.hidden_states = extract_hidden_states(outputs.hidden_states,
            start_index, end_index)
    if hasattr(outputs, 'attentions'):
        o.attentions = extract_attentions(outputs.attentions,
            start_index, end_index)
    return o

def extract_hidden_states(hidden_states, start_index, end_index):
    '''extract the hidden states that overlap with the start and end indices
    '''
    if hidden_states is None:
        return None
    hs = []
    for hidden_state in hidden_states:
        if hidden_state is None:
            hs.append(None)
        else:
            hs.append(hidden_state[:,start_index:end_index,:])
    return hs

def extract_attentions(attentions, start_index, end_index):
    '''extract the attentions that overlap with the start and end indices'''
    if attentions is None:
        return None
    extracted = []
    for attention in attentions:
        if attention is None:
            extracted.append(None)
        else:
            extracted.append(
                attention[:,:,start_index:end_index,start_index:end_index]
            )
    return extracted

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

def determine_n_frames_from_outputs(outputs):
    '''determine the number of frames from available model outputs'''
    frame_counts = []
    o = outputs
    if hasattr(o, 'extract_features') and o.extract_features is not None:
        frame_counts.append(o.extract_features.shape[1])

    if hasattr(o, 'hidden_states') and o.hidden_states is not None:
        hs = o.hidden_states
        hs_counts = [hidden_state.shape[1] for hidden_state in hs
            if hidden_state is not None]
        if hs_counts:
            if len(set(hs_counts)) != 1:
                m = 'hidden_states do not have matching frame counts\n'
                raise ValueError(f'{m}, counts: {hs_counts}')
            frame_counts.append(hs_counts[0])

    if hasattr(o, 'attentions') and o.attentions is not None:
        a = o.attentions
        a_counts = [x.shape[2] for x in a if x is not None]
        if a_counts:
            if len(set(a_counts)) != 1:
                m = 'attentions do not have matching frame counts'
                raise ValueError(f'{m}, counts: {a_counts}')
            frame_counts.append(a_counts[0])
    if not frame_counts:
        raise ValueError('No frame-bearing outputs available')
    if len(set(frame_counts)) != 1:
        attrs = [x for x in ['extract_features', 'hidden_states', 'attentions'] 
            if hasattr(o, x) and getattr(o, x) is not None]
        attr_counts = {attr: count for attr, count in zip(attrs, frame_counts)}
        m = 'Output frame counts do not match'
        raise ValueError(f'{m}, counts: {attr_counts}')
    return frame_counts[0]
