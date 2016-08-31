#!/usr/bin/env python
# encoding: utf-8
"""
DownBeatTracker2 (down-)beat detection algorithm.

"""

from __future__ import absolute_import, division, print_function

import argparse
import warnings
from madmom.processors import (IOProcessor, io_arguments, ParallelProcessor)
from madmom.utils import search_files, match_file
from madmom.features import ActivationsProcessor
from madmom.features.downbeats import (DBNBarTrackingProcessor,
                                       LoadBeatsProcessor)


def match_files(files, input_suffix, beat_suffix):
    """
    Find all matching pairs of audio/feature files and beat file

    :param files:               list of filenames
    :param input_suffix:        suffix of input files
    :param beat_suffix:         suffix of beat files
    :return matched_input_files:list of input files
    :return matched_beat_files: list of beat files

    """
    matched_input_files = []
    matched_beat_files = []
    input_files = search_files(files, input_suffix)
    beat_files = search_files(files, beat_suffix)
    # check if each input file has a match in beat_files
    for num_file, in_file in enumerate(input_files):
        matches = match_file(in_file, beat_files, input_suffix, beat_suffix)
        if len(matches) > 1:
            # exit if multiple detections were found
            raise SystemExit("multiple beat annotations for %s "
                             "found" % in_file)
        elif len(matches) == 0:
            # output a warning if no detections were found
            warnings.warn(" can't find beat detections for %s" % in_file)
            continue
        else:
            # use the first (and only) matched detection file
            matched_input_files.append(in_file)
            matched_beat_files.append(matches[0])
    return matched_input_files, matched_beat_files


def main():
    """DownBeatTracker"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The DownBeatTracker program detects all (down-)beats in an audio file
    according to the method described in:

    This program can be run in 'single' file mode to process a single audio
    file and write the detected beats to STDOUT or the given output file.

    $ DownBeatTracker single INFILE [-o OUTFILE]

    If multiple audio files should be processed, the program can also be run
    in 'batch' mode to save the detected beats to files with the given suffix.

    $ DownBeatTracker batch [-o OUTPUT_DIR] [-s OUTPUT_SUFFIX] LIST OF FILES

    If no output directory is given, the program writes the files with the
    detected beats to same location as the audio files.

    The 'pickle' mode can be used to store the used parameters to be able to
    exactly reproduce experiments.

    ''')
    # version
    p.add_argument('--version', action='version',
                   version='BarTracker.2016')
    p.add_argument('-lb', dest='load_beats', action='store_true',
                   default=True, help='load beats from file [default=%('
                                      'default).s]')
    p.add_argument('-bs', dest='beat_suffix', type=str, default='.beats',
                   help='suffix of beat annotation files '
                   '[default=%(default).s]', action='store')
    p.add_argument('-is', dest='input_suffix', type=str, default='.npy',
                   help='suffix of input files [default=%(default).s]',
                   action='store')
    p.add_argument('-pattern_change_prob', type=float, default=0.0,
                   help='pattern change probability [default=%(default).d]',
                   action='store')
    # TODO: switch warnings on again
    warnings.filterwarnings("ignore")
    # add processor arguments
    io_arguments(p, output_suffix='.beats.txt')
    ActivationsProcessor.add_arguments(p)
    DBNBarTrackingProcessor.add_arguments(p)
    # parse arguments
    args = p.parse_args()
    # load activations and corresponding beat times
    rnn_activation_loader = ActivationsProcessor(mode='r', fps=-1, **vars(
        args))
    # match feature and beat files
    args.files, beat_files = match_files(args.files, args.input_suffix,
                                         args.beat_suffix)
    beat_loader = LoadBeatsProcessor(beat_files, args.beat_suffix)
    input_hmm = ParallelProcessor([rnn_activation_loader, beat_loader])

    # downbeat processor
    downbeat_processor = DBNBarTrackingProcessor(**vars(args))
    if args.downbeats:
        # simply write the timestamps
        from madmom.utils import write_events as writer
    else:
        # borrow the note writer for outputting timestamps + beat numbers
        from madmom.features.notes import write_notes as writer
        # sequentially process them
    out_processor = [downbeat_processor, writer]

    # create an IOProcessor
    processor = IOProcessor(input_hmm, out_processor)

    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == '__main__':
    main()
