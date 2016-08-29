# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.features.beats module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from . import AUDIO_PATH, ANNOTATIONS_PATH
from madmom.audio.chroma import CLPChroma
from madmom.features.downbeats import *
from os.path import join as pj


sample_file = pj(AUDIO_PATH, "sample.wav")
sample_beats = pj(ANNOTATIONS_PATH, "sample.beats")


class TestBeatSyncProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = BeatSyncProcessor(beat_subdivision=2,
                                           sum_func=np.mean, fps=100)

    def test_process(self):
        data = list([CLPChroma(sample_file, fps=100)])
        data.append(np.loadtxt(sample_beats))
        feat_sync = self.processor.process(data)
        target = np.array([0.26559311, 0.14897873, 0.22903995, 0.4171332,
                           0.1598738, 0.22344749, 0.14074484, 0.16701094,
                           0.60201389, 0.24119368, 0.23549884, 0.21930203,
                           0.24890325, 0.13355654, 0.20016374, 0.47378265,
                           0.18951703, 0.16960049, 0.13912612, 0.18324906,
                           0.60975746, 0.19958183, 0.17585503, 0.24287562])
        self.assertTrue(np.allclose(feat_sync[0, :], target, rtol=1e-1))


class TestRNNBarTrackingProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = RNNBarProcessor()

    def test_process(self):
        # check RNN activations and feature computation
        params = {'load_beats': True, 'beat_files': list([sample_beats]),
                  'beat_suffix': '.beats'}
        activations, beats = self.processor.process(sample_file, params)
        target = np.array([0.48208269, 0.12524545, 0.1998145])
        self.assertTrue(np.allclose(activations, target, rtol=1e-2))


class TestDBNBarTrackingProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = DBNBarTrackingProcessor()

    def test_process(self):
        # check DBN output
        in_act = np.array([0.48208269, 0.12524545, 0.1998145])
        beats = np.array([0.0913, 0.7997, 1.4806, 2.1478])
        downbeats = self.processor.process(list([in_act, beats]))
        target = np.array([[0.0913, 1.],
                           [0.7997, 2.],
                           [1.4806, 3.],
                           [2.1478, 1.]])
        self.assertTrue(np.allclose(downbeats, target))
        path, log = self.processor.hmm.viterbi(in_act)
        self.assertTrue(np.allclose(log, -12.222513053716115))

