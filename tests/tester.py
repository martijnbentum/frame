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
    return SimpleNamespace(
        extract_features = extract_features,
        hidden_states = hidden_states,
        audio_filename = 'dummy.wav',
        identifier = 'dummy-id',
        name = 'dummy-name',
    )


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
