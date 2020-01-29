#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2019 Quentin Peter

This file is part of WF_NTP.

WF_NTP is distributed under CC BY-NC-SA version 4.0. You should have
recieved a copy of the licence along with WF_NTP. If not, see
https://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

DEFAULTS = {
    'video_filename': ('', str),
    'start_frame': (0, int),
    'limit_images_to': (None, int),
    'fps': (20, float),
    'px_to_mm': (0.04, float),
    'darkfield': (False, bool),

    'keep_paralyzed_method': (False, bool),
    'std_px': (64, int),
    'threshold': (9, int),
    'opening': (1, int),
    'closing': (3, int),
    'prune_size': (0, int),
    'skeletonize': (False, bool),
    'do_full_prune': (False, bool),

    'min_size': (25, int),
    'max_size': (120, int),
    'minimum_ecc': (0.93, float),

    'use_average': (True, bool),
    'lower': (0, int),
    'upper': (100, int),
    'Bends_max': (20, float),
    'Speed_max': (0.035, float),
    'extra_filter': (False, bool),
    'cutoff_filter': (False, bool),

    'max_dist_move': (10, int),
    'min_track_length': (50, int),
    'memory': (5, int),

    'bend_threshold': (2.1, float),
    'minimum_bends': (0, float),
    'frames_to_estimate_velocity': (49, int),

    'maximum_bpm': (0.5, float),
    'maximum_velocity': (0.1, float),

    'regions': ([], list),

    'save_as': ('', str),
    'output_overlayed_images': (0, int),
    'font_size': (8, int),
    'scale_bar_size': (1.0, float),
    'scale_bar_thickness': (7, int),

    'max_plot_pixels': (1500**2, int),
    'Z_skip_images': (1, int),
    'use_images': (100, int),
    'use_around': (5, int),
     }