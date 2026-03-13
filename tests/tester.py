import unittest
from types import SimpleNamespace

import numpy as np

from frame.dummy_data import generate_segments
from frame.frames import determine_n_frames_from_outputs
from frame.frames import extract_outputs_times
from frame.frames import make_frames_from_duration
from frame.frames import make_frames_from_outputs
from frame.labels_to_frames import Data


def make_dummy_outputs(n_frames = 4, feature_size = 3):
    extract_features = np.arange(n_frames * feature_size, dtype = float)
    extract_features = extract_features.reshape(1, n_frames, feature_size)
    hidden_states = [
        np.arange(n_frames * feature_size, dtype = float).reshape(
            1, n_frames, feature_size
        )
    ]
    attentions = [
        np.arange(n_frames * n_frames, dtype = float).reshape(
            1, 1, n_frames, n_frames
        )
    ]
    return SimpleNamespace(
        extract_features = extract_features,
        hidden_states = hidden_states,
        attentions = attentions,
        audio_filename = 'dummy.wav',
        identifier = 'dummy-id',
        name = 'dummy-name',
    )


def make_attention_outputs(n_frames = 4, with_batch_dim = False):
    outputs = make_dummy_outputs(n_frames = n_frames)
    attention = np.arange(2 * n_frames * n_frames, dtype = float).reshape(
        2, n_frames, n_frames
    )
    if with_batch_dim:
        attention = attention.reshape(1, 2, n_frames, n_frames)
    outputs.attentions = [attention]
    return outputs


class FramesRegressionTests(unittest.TestCase):
    def test_make_frames_from_outputs_keeps_outputs_attached(self):
        outputs = make_dummy_outputs()
        frames = make_frames_from_outputs(outputs)

        self.assertIs(frames.outputs, outputs)
        np.testing.assert_array_equal(frames.cnn(), outputs.extract_features[0])

    def test_determine_n_frames_from_outputs_accepts_matching_sources(self):
        outputs = make_dummy_outputs(n_frames = 4)
        outputs.attentions = [
            np.zeros((1, 2, 4, 4), dtype = float),
            np.ones((1, 2, 4, 4), dtype = float),
        ]

        self.assertEqual(determine_n_frames_from_outputs(outputs), 4)

    def test_determine_n_frames_from_outputs_rejects_mismatched_sources(self):
        outputs = make_dummy_outputs(n_frames = 4)
        outputs.attentions = [np.zeros((1, 2, 5, 5), dtype = float)]

        with self.assertRaises(ValueError):
            determine_n_frames_from_outputs(outputs)

    def test_make_frames_from_outputs_accepts_none_hidden_state_layers(self):
        outputs = make_dummy_outputs(n_frames = 4)
        outputs.hidden_states = [None, outputs.hidden_states[0]]

        frames = make_frames_from_outputs(outputs)

        self.assertEqual(frames.transformer_none_indices, [0])
        self.assertEqual(frames.transformer_available_indices, [1])
        self.assertIsNone(frames.frames[0].transformer(0))
        np.testing.assert_array_equal(
            frames.frames[0].transformer(1),
            outputs.hidden_states[1][0, 0, :],
        )

    def test_frame_attention_exposes_query_key_relation(self):
        outputs = make_attention_outputs(n_frames = 4)
        frames = make_frames_from_outputs(outputs)

        np.testing.assert_array_equal(
            frames.frames[1].attention_query(0),
            outputs.attentions[0][:, 1, :],
        )
        np.testing.assert_array_equal(
            frames.frames[2].attention_key(0),
            outputs.attentions[0][:, :, 2],
        )
        np.testing.assert_array_equal(
            frames.frames[1].attention_query_key(0, 3),
            outputs.attentions[0][:, 1, 3],
        )

    def test_frame_attention_accepts_batch_dimension(self):
        outputs = make_attention_outputs(n_frames = 4, with_batch_dim = True)
        frames = make_frames_from_outputs(outputs)

        np.testing.assert_array_equal(
            frames.frames[1].attention_query(0),
            outputs.attentions[0][0, :, 1, :],
        )
        np.testing.assert_array_equal(
            frames.frames[2].attention_key(0),
            outputs.attentions[0][0, :, :, 2],
        )

    def test_frames_attention_query_key_supports_selection(self):
        outputs = make_attention_outputs(n_frames = 4)
        frames = make_frames_from_outputs(outputs)

        np.testing.assert_array_equal(
            frames.attention_query(0, position = 'middle'),
            outputs.attentions[0][:, 1, :],
        )
        np.testing.assert_array_equal(
            frames.attention_query_key(0, 2, position = 'middle'),
            outputs.attentions[0][:, 1, 2],
        )

    def test_select_frames_handles_missing_end_time_and_overlap_filter(self):
        outputs = make_dummy_outputs(n_frames = 5)
        frames = make_frames_from_outputs(outputs)

        selected = frames.select_frames(start_time = 0.02, end_time = None)
        self.assertEqual(selected[0].index, 0)
        self.assertEqual(selected[-1].index, 4)

        overlap_selected = frames.select_frames(
            start_time = 0.02,
            end_time = 0.05,
            percentage_overlap = 50,
        )
        self.assertEqual([frame.index for frame in overlap_selected], [1])

    def test_extract_outputs_times_uses_valid_constructor(self):
        outputs = make_dummy_outputs(n_frames = 5)

        extracted = extract_outputs_times(outputs, start_time = 0.02, end_time = 0.05)

        self.assertEqual(extracted.extract_features.shape[1], 2)
        self.assertEqual(extracted.attentions[0].shape[2:], (2, 2))
        self.assertEqual(extracted.start_time, 0.02)

    def test_make_frames_from_duration_supports_short_segments(self):
        for duration in [0.001, 0.01, 0.02, 0.025]:
            frames = make_frames_from_duration(duration)
            self.assertGreaterEqual(frames.n_frames, 1)


class DataRegressionTests(unittest.TestCase):
    def test_data_str_handles_edge_targets(self):
        for target_index in [0, 1, 2]:
            labels = generate_segments(n = 3, start = 0.0, duration = 0.2)
            data = Data(labels, target_index, audio_duration = 2.0)

            rendered = str(data)

            self.assertIn('preceding frames:', rendered)
            self.assertIn('following frames:', rendered)


if __name__ == '__main__':
    unittest.main()
