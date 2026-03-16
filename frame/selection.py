import numpy as np

def make_frame_selection_from_segment(frames, segment,
    percentage_overlap = None, label = None):
    '''Create a FrameSelection from a single time-aligned segment.
    frames              Frames object to select from
    segment             object with start and end times
    percentage_overlap  percentage of frame duration that must overlap
    label               optional label for the selection
    '''
    start, end = segment_to_start_end(segment)
    indices = segment_to_frame_indices(frames, segment,
        percentage_overlap = percentage_overlap)
    if label is None:
        label = segment_to_label(segment)
    return FrameSelection(frames, indices, label = label, start = start,
        end = end)


class FrameSelection:
    '''View on a subset of frames from one Frames object.
    frames              parent Frames object
    indices             selected frame indices
    label               optional label for this selection
    start               optional start time in seconds
    end                 optional end time in seconds
    '''

    def __init__(self, frames, indices, label = '', start = None,
        end = None):
        self.frames = frames
        self.indices = sorted(set(int(i) for i in indices))
        self.label = label
        self.start = start
        self.end = end

    @classmethod
    def from_segment(cls, frames, segment, percentage_overlap = None,
        label = None):
        '''Create a selection from a segment with start and end times.
        frames              Frames object to select from
        segment             object with start and end times
        percentage_overlap  percentage of frame duration that must overlap
        label               optional label for the selection
        '''
        return make_frame_selection_from_segment(frames, segment,
            percentage_overlap = percentage_overlap, label = label)

    def __repr__(self):
        return f'FrameSelection(label={self.label!r}, n_frames={len(self)})'

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        for index in self.indices:
            yield self.frames[index]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return FrameSelection(self.frames, self.indices[index],
                label = self.label, start = self.start, end = self.end)
        return self.frames[self.indices[index]]

    @property
    def frame_objects(self):
        '''Return selected frame objects.'''
        return [self.frames[index] for index in self.indices]

    @property
    def n_frames(self):
        '''Return number of selected frames.'''
        return len(self.indices)

    def _validate_nonempty(self):
        '''Check whether selection contains frames.'''
        if not self.indices:
            raise ValueError('frame selection is empty')

    def _stack_attention(self, getter):
        '''Collect attention arrays for all frames in the selection.
        getter              callable applied to each frame
        '''
        self._validate_nonempty()
        return np.stack([getter(frame) for frame in self], axis = 0)

    def attention_to(self, layer, head = None, aggregate = None):
        '''Return attention from selected frames to all frames.
        layer               transformer layer index
        head                optional head index
        aggregate           None, mean, or sum over selected frames
        '''
        attention = self._stack_attention(
            lambda frame: frame.attention_to(layer, head = head))
        if head is None:
            attention = np.transpose(attention, (1, 0, 2))
        if aggregate is None:
            return attention
        if aggregate == 'mean':
            return attention.mean(axis = -2)
        if aggregate == 'sum':
            return attention.sum(axis = -2)
        raise ValueError("aggregate must be None, 'mean', or 'sum'")

    def attention_from(self, layer, head = None, aggregate = None):
        '''Return attention to selected frames from all frames.
        layer               transformer layer index
        head                optional head index
        aggregate           None, mean, or sum over selected frames
        '''
        attention = self._stack_attention(
            lambda frame: frame.attention_from(layer, head = head))
        if head is None:
            attention = np.transpose(attention, (1, 2, 0))
        else:
            attention = np.transpose(attention, (1, 0))
        if aggregate is None:
            return attention
        if aggregate == 'mean':
            return attention.mean(axis = -1)
        if aggregate == 'sum':
            return attention.sum(axis = -1)
        raise ValueError("aggregate must be None, 'mean', or 'sum'")

    def attention_query(self, layer, head = None, aggregate = None):
        '''Backward-compatible alias for attention_to.
        layer               transformer layer index
        head                optional head index
        aggregate           None, mean, or sum over selected frames
        '''
        return self.attention_to(layer, head = head, aggregate = aggregate)

    def attention_key(self, layer, head = None, aggregate = None):
        '''Backward-compatible alias for attention_from.
        layer               transformer layer index
        head                optional head index
        aggregate           None, mean, or sum over selected frames
        '''
        return self.attention_from(layer, head = head, aggregate = aggregate)

    def attention_to_selection(self, other, layer, head = None,
        aggregate = 'mass'):
        '''Return attention from this selection to another selection.
        other               target FrameSelection
        layer               transformer layer index
        head                optional head index
        aggregate           mass, mean, or max
        '''
        other._validate_nonempty()
        block = self.attention_to(layer, head = head)[..., other.indices]
        if aggregate == 'mass':
            return self._aggregate_mass(block)
        if aggregate == 'mean':
            return block.mean(axis = (-2, -1))
        if aggregate == 'max':
            return block.max(axis = (-2, -1))
        raise ValueError("aggregate must be 'mass', 'mean', or 'max'")

    def attention_from_selection(self, other, layer, head = None,
        aggregate = 'mass'):
        '''Return attention from another selection to this selection.
        other               source FrameSelection
        layer               transformer layer index
        head                optional head index
        aggregate           mass, mean, or max
        '''
        return other.attention_to_selection(self, layer = layer, head = head,
            aggregate = aggregate)

    def attention_matrix(self, selections, layer, head = None,
        aggregate = 'mass'):
        '''Compute a selection-by-selection attention matrix.
        selections          list of FrameSelection objects
        layer               transformer layer index
        head                optional head index
        aggregate           mass, mean, or max
        '''
        items = [self] + list(selections)
        matrix = np.zeros((len(items), len(items)), dtype = float)
        for i, source in enumerate(items):
            for j, target in enumerate(items):
                value = source.attention_to_selection(target, layer = layer,
                    head = head, aggregate = aggregate)
                if np.ndim(value) > 0:
                    value = np.mean(value)
                matrix[i, j] = value
        return matrix

    def _aggregate_mass(self, block):
        '''Aggregate block as average query-wise attention mass.
        block               attention block with shape (heads, Q, K) or (Q, K)
        '''
        block = np.asarray(block)
        if block.ndim == 3:
            return block.sum(axis = -1).mean(axis = -1)
        if block.ndim == 2:
            return block.sum(axis = -1).mean()
        raise ValueError('expected block with shape (heads, Q, K) or (Q, K)')

    def mean_time(self):
        '''Return mean time of selected frames.'''
        times = [(frame.start_time + frame.end_time) / 2 for frame in self]
        if not times:
            return None
        return float(np.mean(times))

    def to_dict(self):
        '''Return simple dictionary summary.'''
        return {
            'label': self.label,
            'start': self.start,
            'end': self.end,
            'indices': list(self.indices),
            'n_frames': len(self.indices),
        }



def segment_to_frame_indices(frames, segment, percentage_overlap = None):
    '''Return frame indices that overlap with a segment interval.
    frames              Frames object to select from
    segment             object with start and end times
    percentage_overlap  percentage of frame duration that must overlap
    '''
    start, end = segment_to_start_end(segment)
    selected_frames = frames.select_frames(start_time = start,
        end_time = end, percentage_overlap = percentage_overlap)
    return [frame.index for frame in selected_frames]


def segment_to_start_end(segment):
    '''Extract start and end times from a segment-like object.
    segment             object with start and end times
    '''
    start = _segment_value(segment, 'start', 'start_time', 'start_seconds')
    end = _segment_value(segment, 'end', 'end_time', 'end_seconds')
    if start is None or end is None:
        raise ValueError('segment must expose start and end times')
    return float(start), float(end)


def segment_to_label(segment):
    '''Extract a label from a segment-like object if present.
    segment             object that may expose a label
    '''
    label = _segment_value(segment, 'label', 'name', 'text')
    if label is None:
        return ''
    return str(label)


def _segment_value(segment, *names):
    '''Return the first available attribute on a segment.
    segment             object that may expose the requested attributes
    names               ordered attribute names to try
    '''
    for name in names:
        if hasattr(segment, name):
            return getattr(segment, name)
    return None
