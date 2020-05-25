"""
Copyright (C) 2019 Quentin Peter

This file is part of WF_NTP.

WF_NTP is distributed under CC BY-NC-SA version 4.0. You should have
recieved a copy of the licence along with WF_NTP. If not, see
https://creativecommons.org/licenses/by-nc-sa/4.0/.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import interpolate, ndimage
import cv2
import os
import time
import mahotas as mh
import pandas as pd
import trackpy as tp
from skimage import measure, morphology, io
import skimage.draw
import pickle
import warnings
import matplotlib.path as mplPath
from collections import defaultdict, Counter
from skimage.transform import resize
import traceback
from scipy.signal import savgol_filter
import functools
import sys
import json


class StdoutRedirector(object):
    """File that writes to a queue with a prefix."""

    def __init__(self, queue, prefix=None):
        self.queue = queue
        if not prefix:
            prefix = ''
        self.prefix = prefix

    def write(self, string):
        """When writing, add to the queue."""
        self.queue.put(self.prefix + string)

    def flush(self):
        """There is no flush."""
        pass


def save_settings(settings):
    # Make output directory
    try:
        os.mkdir(settings['save_as'])
    except OSError:
        print(
            'Warning: job folder "%s" already created, overwriting.' %
            settings['save_as'])

    settingsfilename = os.path.join(settings['save_as'], 'settings.json')
    with open(settingsfilename, 'w') as f:
        json.dump(settings, f, indent=4)


def run_tracker(settings, stdout_queue=None):
    """
    Run the tracker with the given settings.

    stdout_queue can be used to redirect stdout.
    """
    if stdout_queue:
        sys.stdout = StdoutRedirector(stdout_queue, settings["stdout prefix"])
        sys.stderr = StdoutRedirector(stdout_queue, settings["stdout prefix"])

    save_settings(settings)

    # Do some adjustments
    settings = settings.copy()
    settings["frames_to_estimate_velocity"] = min([
        settings["frames_to_estimate_velocity"],
        settings["min_track_length"]])
    settings["bend_threshold"] /= 100.

    video = Video(settings, grey=True)

    print('Video shape:', video[0].shape)

    regions = settings["regions"]
    try:
        len(regions)
    except Exception:
        regions = {}
    if len(regions) == 0:
        im = np.ones_like(video[0])
        all_regions = im > 0.1
    else:
        all_regions = np.zeros_like(video[0])
        for key, d in list(regions.items()):
            im = np.zeros_like(video[0])
            rr, cc = skimage.draw.polygon(np.array(d['y']), np.array(d['x']))
            try:
                im[rr, cc] = 1
            except IndexError:
                print('Region "', key, '" cannot be applied to video',
                      settings["video_filename"])
                print('Input image sizes do not match.')
                return None, None
            all_regions += im
        all_regions = all_regions > 0.1
    settings["all_regions"] = all_regions
    settings["regions"] = regions

    t0 = time.time()
    save_folder = settings["save_as"]
    ims_folder = os.path.join(save_folder, 'imgs')
    if not os.path.exists(ims_folder):
        os.mkdir(ims_folder)

    # Analysis
    print_data, locations = track_all_locations(video, settings, stdout_queue)

    if settings["stop_after_example_output"]:
        return print_data, None
    track = form_trajectories(locations, settings)

    results = extract_data(track, settings)
    if not check_for_worms(results["particle_dataframe"].index,
                           settings):
        print('No worms detected. Stopping!')
        return print_data, None
    # Output
    write_results_file(results, settings)

    print('Done (in %.1f minutes).' % ((time.time() - t0) / 60.))
    video.release()
    return print_data, results['particle_dataframe'].loc[:, "bends"]


class Video:
    """Class to read a video frame by frame instead of loading in memory."""

    def __init__(self, settings, grey=False):
        video_filename = settings["video_filename"]
        self.video_filename = video_filename
        self.cap = None
        if not os.path.exists(video_filename):
            raise RuntimeError(f"{video_filename} does not exist.")

        self.cap = cv2.VideoCapture(video_filename)
        self.len = (self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    - settings["start_frame"])
        self.start_frame = settings["start_frame"]
        limit_images_to = settings["limit_images_to"]
        if (limit_images_to and limit_images_to < (
                self.len - self.start_frame)):
            self.len = limit_images_to
        self.grey = grey
        if grey:
            # Check we only have two dims
            for _ in range(100):
                ret, frame = self.cap.read()
                if ret:
                    break
            if len(frame.shape) == 2:
                self.grey = False
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def __next__(self):
        ret, frame = self.cap.read()
        if ret:
            if self.grey:
                return frame[:, :, 0]
            else:
                return frame
        else:
            raise StopIteration

    def set_index(self, i):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)

    def restart(self):
        self.set_index(self.start_frame)

    def __getitem__(self, i):
        if i < 0:
            i += self.len
        self.set_index(self.start_frame + i)
        return next(self)

    def __len__(self):
        return int(self.len)

    def __del__(self):
        if self.cap:
            self.cap.release()

    def release(self):
        self.cap.release()
        self.cap = None


def track_all_locations(video, settings, stdout_queue):
    """Track and get all locations."""
    def get_Z_brightness(zi):
        if settings["keep_paralyzed_method"]:
            return find_Z_with_paralyzed(video, settings, *zi)
        else:
            return find_Z(video, settings, *zi)

    apply_indeces = list(
        map(int, list(np.linspace(0, len(video),
                                  len(video) // settings["use_images"] + 2))))
    apply_indeces = list(zip(apply_indeces[:-1], apply_indeces[1:]))
    Z_indeces = [(max([0, i - settings["use_around"]]),
                  min(j + settings["use_around"], len(video)))
                 for i, j in apply_indeces]

    # Get frames0 print material
    Z, mean_brightness = get_Z_brightness(Z_indeces[0])
    print_data = process_frame(settings, Z, mean_brightness,
                               len(video),
                               args=(0, video[0]),
                               return_plot=True)

    if settings["stop_after_example_output"]:
        return print_data, None

    # Process all frames
    args = list(zip(apply_indeces, Z_indeces))

    def locate(args):
        i, zi = args
        Z, mean_brightness = get_Z_brightness(zi)
        return process_frames(video, settings, *i, Z=Z,
                              mean_brightness=mean_brightness)

    split_results = list(map(locate, args))
    locations = []
    for l in split_results:
        locations += l
    return print_data, locations


def process_frame(settings, Z, mean_brightness, nframes,
                  args=None, return_plot=False):
    """Locate worms in a given frame."""
    i, frameorig = args
    print(' : Locating in frame %i/%i' % (i + 1 + settings["start_frame"],
                                          nframes + settings["start_frame"]))

    if mean_brightness:
        frame = frameorig * mean_brightness / np.mean(frameorig)
    else:
        frame = np.array(frameorig, dtype=np.float64)
    frame = np.abs(frame - Z) * settings["all_regions"]
    if (frame > 1.1).any():
        frame /= 255.

    thresholded = frame > (settings["threshold"] / 255.)
    opening = settings["opening"]
    closing = settings["closing"]
    save_folder = settings["save_as"]
    if opening > 0:
        frame_after_open = ndimage.binary_opening(
            thresholded,
            structure=np.ones((opening, opening))).astype(np.int)
    else:
        frame_after_open = thresholded

    if closing > 0:
        frame_after_close = ndimage.binary_closing(
            frame_after_open,
            structure=np.ones((closing, closing))).astype(np.int)
    else:
        frame_after_close = frame_after_open

    labeled, _ = mh.label(frame_after_close, np.ones(
        (3, 3), bool))
    sizes = mh.labeled.labeled_size(labeled)

    remove = np.where(np.logical_or(sizes < settings["min_size"],
                                    sizes > settings["max_size"]))
    labeled_removed = mh.labeled.remove_regions(labeled, remove)
    labeled_removed, n_left = mh.labeled.relabel(labeled_removed)

    props = measure.regionprops(labeled_removed, coordinates='xy')
    prop_list = [{"area": props[j].area, "centroid":props[j].centroid,
                  "eccentricity":props[j].eccentricity,
                  "area_eccentricity":props[j].eccentricity,
                  "minor_axis_length":props[j].minor_axis_length /
                  (props[j].major_axis_length + 0.001)}
                 for j in range(len(props))]
    if settings["skeletonize"]:
        skeletonized_frame = morphology.skeletonize(frame_after_close)
        skeletonized_frame = prune(skeletonized_frame,
                                   settings["prune_size"])

        skel_labeled = labeled_removed * skeletonized_frame
        if settings["do_full_prune"]:
            skel_labeled = prune_fully(skel_labeled)

        skel_props = measure.regionprops(skel_labeled, coordinates='xy')
        for j in range(len(skel_props)):
            prop_list[j]["length"] = skel_props[j].area
            prop_list[j]["eccentricity"] = skel_props[j].eccentricity
            prop_list[j]["minor_axis_length"] = \
                skel_props[j].minor_axis_length\
                / (skel_props[j].major_axis_length + 0.001)

    if return_plot:
        return (sizes, save_folder, frameorig, Z, frame, thresholded,
                frame_after_open, frame_after_close, labeled, labeled_removed,
                (skel_labeled if settings["skeletonize"] else None))

    output_overlayed_images = settings["output_overlayed_images"]
    if i < output_overlayed_images or output_overlayed_images is None:
        io.imsave(os.path.join(save_folder, "imgs", '%05d.jpg' % (i)),
                  np.array(255 * (labeled_removed == 0), dtype=np.uint8),
                  check_contrast=False)

    return prop_list


def process_frames(video, settings, i0, i1, Z, mean_brightness):
    """Frocess frames from i0 to i1."""
    func = functools.partial(
        process_frame, settings, Z, mean_brightness, len(video))

    def args():
        for i in range(i0, i1):
            yield i, video[i]

    return map(func, args())


def form_trajectories(loc, settings):
    """Form worm trajectories."""
    print('Forming worm trajectories...', end=' ')
    data = {'x': [], 'y': [], 'frame': [],
            'eccentricity': [], 'area': [],
            'minor_axis_length': [],
            'area_eccentricity': []}
    for t, l in enumerate(loc):
        data['x'] += [d['centroid'][0] for d in l]
        data['y'] += [d['centroid'][1] for d in l]
        data['eccentricity'] += [d['eccentricity'] for d in l]
        data['area_eccentricity'] += [d['area_eccentricity'] for d in l]
        data['minor_axis_length'] += [d['minor_axis_length'] for d in l]
        data['area'] += [d['area'] for d in l]
        data['frame'] += [t] * len(l)
    data = pd.DataFrame(data)
    try:
        track = tp.link_df(data, search_range=settings["max_dist_move"],
                           memory=settings["memory"])
    except tp.linking.SubnetOversizeException:
        raise RuntimeError(
            'Linking problem too complex.'
            ' Reduce maximum move distance or memory.')
    track = tp.filter_stubs(track, min([settings["min_track_length"],
                                        len(loc)]))
    try:
        with open(os.path.join(settings["save_as"], 'track.p'),
                  'bw') as trackfile:
            pickle.dump(track, trackfile)
    except Exception:
        traceback.print_exc()
        print('Warning: no track file saved. Track too long.')
        print('         plot_path.py will not work on this file.')

    return track


def extract_data(track, settings):
    """Extract data from track and return a pandas DataFrame."""
    P = track['particle']
    columns_dtype = {
        "bends": object
    }
    # Use particle as index
    particle_dataframe = pd.DataFrame(index=P.unique(),
                                      columns=columns_dtype.keys())
    # Set non float dtype correctly
    particle_dataframe = particle_dataframe.astype(columns_dtype)

    T = track['frame']
    X = track['x']
    Y = track['y']

    regions = settings["regions"]
    if len(regions) > 1:
        reg_paths = make_region_paths(regions)

    drop_list = []
    for p in particle_dataframe.index:
        # Define signals
        t = T[P == p]
        ecc = track['eccentricity'][P == p]
        area_ecc = track['area_eccentricity'][P == p]
        # mal = track['minor_axis_length'][P == p]
        area = track['area'][P == p]

        window_size = 7

        # Smooth bend signal
        x = np.arange(min(t), max(t) + 1)
        f = interpolate.interp1d(t, ecc)
        y = f(x)
        max_window_size = 2 * ((len(y) + 1) // 2) - 1
        smooth_y = savgol_filter(y, min(window_size, max_window_size), 2)

        # Use eccentricity of non-skeletonized to filter worm-like
        f = interpolate.interp1d(t, area_ecc)
        y = f(x)
        max_window_size = 2 * ((len(y) + 1) // 2) - 1
        area_ecc = savgol_filter(y, min(window_size, max_window_size), 2)

        # Interpolate circle-like worms
        # (these are removed later if count is low)
        idx = area_ecc > settings["minimum_ecc"]
        if sum(idx) > 0:
            smooth_y = np.interp(x, x[idx], smooth_y[idx])
            particle_dataframe.at[p, "Round ratio"] = (
                1.0 - float(sum(idx)) / float(len(idx)))
        else:
            # 0.001,0.991,0.992 are dummy variables specifically picked
            # to deal with coilers, see protocol.
            lengthX = 0.001 / len(idx)
            smooth_y = np.arange(0.991, 0.992, lengthX)
            np.random.shuffle(smooth_y)
            particle_dataframe.at[p, "Round ratio"] = (
                1.0 - float(sum(idx)) / float(len(idx)))

        # Bends
        bend_times = extract_bends(x, smooth_y, settings)
        if len(bend_times) < settings["minimum_bends"]:
            drop_list.append(p)
            continue
        bl = form_bend_array(bend_times, T[P == p])
        if len(bl) > 0:
            bl = (np.asarray(bl, float))
        else:
            bl = (np.array([0.0] * len(T[P == p])))

        px_to_mm = settings["px_to_mm"]
        # Area
        if settings["skeletonize"]:
            particle_dataframe.at[p, "Area"] = np.median(area) * px_to_mm
        else:
            particle_dataframe.at[p, "Area"] = np.median(area) * px_to_mm**2

        # Eccentricity
        particle_dataframe.at[p, "eccentricity"] = np.mean(area_ecc)

        # Velocity
        particle_dataframe.at[p, "Speed"] = extract_velocity(
            T[P == p], X[P == p], Y[P == p], settings)

        # Max velocity: 90th percentile to avoid skewed results due to tracking
        # inefficiency
        particle_dataframe.at[p, "Max speed"] = extract_max_speed(
            T[P == p], X[P == p], Y[P == p], settings)

        # Move per bend
        particle_dataframe.at[p, "Dist per bend"] = extract_move_per_bend(
            bl, T[P == p], X[P == p], Y[P == p], px_to_mm)

        particle_dataframe.at[p, "bends"] = bl

        # Sort out low bend number particles
        if bl[-1] < settings["minimum_bends"]:
            drop_list.append(p)

    particle_dataframe.drop(drop_list, inplace=True)

    # BPM
    fps = settings["fps"]
    for index in particle_dataframe.index:
        last_bend = particle_dataframe.at[index, "bends"][-1]
        with warnings.catch_warnings():
            # Ignore ptp warnings as this is a numpy bug
            warnings.simplefilter("ignore")
            particle_dataframe.at[index, "BPM"] = (
                last_bend / np.ptp(T[P == index]) * 60 * fps)
            x = (settings["limit_images_to"] / fps)
            particle_dataframe.at[index, "bends_in_movie"] = (
                last_bend / np.ptp(T[P == index]) * x * fps)
        particle_dataframe.at[index, "Appears in frames"] = len(
            particle_dataframe.at[index, "bends"])

    # Cut off-tool for skewed statistics
    if settings["cutoff_filter"]:
        list_number = []
        frames = []
        for t in set(T):
            if t >= settings["lower"] and t <= settings["upper"]:
                particles_present = len(set(P[T == t]))
                frames.append(t)
                list_number.append(particles_present)

        list_number = np.array(list_number)
        frames = np.array(frames)

        if settings["use_average"]:
            cut_off = int(np.sum(list_number) / len(list_number)) + \
                (np.sum(list_number) % len(list_number) > 0)
        else:
            cut_off = max(list_number)

        # cut off based on selected frames
        original_particles = len(particle_dataframe)

        particle_dataframe = particle_dataframe.head(cut_off)
        removed_particles_cutoff = original_particles - len(particle_dataframe)

        cutoff_filter_data = dict(
            list_number=list_number,
            frames=frames,
            original_particles=original_particles,
            removed_particles_cutoff=removed_particles_cutoff,
            )

    else:
        cutoff_filter_data = None

    # Cut off-tool for boundaries (spurious worms)
    if settings["extra_filter"]:
        mask = (
            (particle_dataframe.loc[:, "BPM"] > settings["Bends_max"]) &
            (particle_dataframe.loc[:, "Speed"] < settings["Speed_max"]))
        extra_filter_spurious_worms = mask.sum()
        particle_dataframe = particle_dataframe.loc[~mask]
    else:
        extra_filter_spurious_worms = None

    region_particles = defaultdict(list)
    for index in particle_dataframe.index:
        # Indetify region
        if len(regions) > 1:
            this_reg = identify_region(X[P == index], Y[P == index], reg_paths)
            if not this_reg:
                continue
        else:
            this_reg = ['all']
        particle_dataframe.at[index, "Region"] = str(this_reg)
        for reg in this_reg:
            region_particles[reg].append(index)

    particle_dataframe.loc[:, "Moving"] = np.logical_or(
        particle_dataframe.loc[:, "BPM"] > settings["maximum_bpm"],
        particle_dataframe.loc[:, "Speed"] > settings["maximum_velocity"])

    return dict(
        cutoff_filter_data=cutoff_filter_data,
        extra_filter_spurious_worms=extra_filter_spurious_worms,
        particle_dataframe=particle_dataframe,
        track=track,
        region_particles=region_particles,
    )


# =============================================================================
# --- Utilities Functions ---
# =============================================================================

def find_Z(video, settings, i0, i1):
    """Get thresholded image."""
    # Adjust brightness:
    frame = video[(i0 + i1) // 2]
    mean_brightness = np.mean(frame)
    if mean_brightness > 1:
        mean_brightness /= 255.
    Z = np.zeros_like(frame, dtype=np.float64)
    if settings["darkfield"]:
        minv = np.zeros_like(frame, dtype=np.float64) + 256
    else:
        minv = np.zeros_like(frame, dtype=np.float64) - 256
    for i in range(i0, i1, settings["Z_skip_images"]):
        frame = video[i]
        frame = frame * mean_brightness / np.mean(frame)
        diff = frame
        if settings["darkfield"]:
            logger = diff < minv
        else:
            logger = diff > minv
        minv[logger] = diff[logger]
        Z[logger] = frame[logger]
    return Z, mean_brightness


def find_Z_with_paralyzed(video, settings, i0, i1):
    """Get thresholded image with paralyzed worms."""
    frame = video[(i0 + i1) // 2]
    Y, X = np.meshgrid(np.arange(frame.shape[1]),
                       np.arange(frame.shape[0]))
    thres = cv2.adaptiveThreshold(
        frame, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 2 * (settings["std_px"] // 2) + 1, 0)
    mask = thres > 0.5
    vals = frame[mask]
    x = X[mask]
    y = Y[mask]
    Z = interpolate.griddata((x, y), vals, (X, Y), method='nearest')
    return Z, False


def find_skel_endpoints(skel):
    """Find skeleton endpoints."""
    skel_endpoints = [
        np.array([[0, 0, 0], [0, 1, 0], [2, 1, 2]]),
        np.array([[0, 0, 0], [0, 1, 2], [0, 2, 1]]),
        np.array([[0, 0, 2], [0, 1, 1], [0, 0, 2]]),
        np.array([[0, 2, 1], [0, 1, 2], [0, 0, 0]]),
        np.array([[2, 1, 2], [0, 1, 0], [0, 0, 0]]),
        np.array([[1, 2, 0], [2, 1, 0], [0, 0, 0]]),
        np.array([[2, 0, 0], [1, 1, 0], [2, 0, 0]]),
        np.array([[0, 0, 0], [2, 1, 0], [1, 2, 0]])]

    ep = 0
    for skel_endpoint in skel_endpoints:
        ep += mh.morph.hitmiss(skel, skel_endpoint)

    return ep


def prune(skel, size):
    """Prune skeleton."""
    for _ in range(size):
        endpoints = find_skel_endpoints(skel)
        skel = np.logical_and(skel, np.logical_not(endpoints))
    return skel


def prune_fully(skel_labeled):
    """Prune skeleton fully."""
    for k in range(1000):
        endpoints = find_skel_endpoints(skel_labeled > 0) > 0
        idx = np.argwhere(endpoints)
        reg = skel_labeled[idx[:, 0], idx[:, 1]]
        count = Counter(reg)
        idx = np.array([idx[i, :] for i in range(len(reg))
                        if count[reg[i]] > 2])
        if len(idx) == 0:
            break
        endpoints[:] = 1
        endpoints[idx[:, 0], idx[:, 1]] = 0
        skel_labeled *= endpoints
    return skel_labeled


def check_for_worms(particles, settings):
    """Check if any worms have been detected."""
    if len(particles) == 0:
        with open(os.path.join(settings["save_as"], 'results.txt'), 'w') as f:
            f.write('---------------------------------\n')
            f.write('    Results for %s \n' % settings["video_filename"])
            f.write('---------------------------------\n\n')
            f.write('No worms detected. Check your settings.\n\n')
        return False
    return True


def make_region_paths(regions):
    reg_paths = {}
    for key, d in list(regions.items()):
        reg_paths[key] = mplPath.Path(
            np.array(list(zip(d['x'] + [d['x'][0]], d['y'] + [d['y'][0]]))))
    return reg_paths


def identify_region(xs, ys, reg_paths):
    regions = {}
    for x, y in zip(xs, ys):
        for key, path in list(reg_paths.items()):
            if path.contains_point((y, x)):
                regions[key] = key
    return list(regions.keys())


def extract_bends(x, smooth_y, settings):
    # Find extrema
    ex = (np.diff(np.sign(np.diff(smooth_y))).nonzero()[0] + 1)
    if len(ex) >= 2 and ex[0] == 0:
        ex = ex[1:]
    bend_times = x[ex]
    bend_magnitude = smooth_y[ex]

    # Sort for extrema satisfying criteria
    idx = np.ones(len(bend_times))
    index = 1
    prev_index = 0
    while index < len(bend_magnitude):
        dist = abs(bend_magnitude[index] - bend_magnitude[prev_index])
        if dist < settings["bend_threshold"]:
            idx[index] = 0
            if index < len(bend_magnitude) - 1:
                idx[index + 1] = 0
            index += 2  # look for next maximum/minimum (not just extrema)
        else:
            prev_index = index
            index += 1
    bend_times = bend_times[idx == 1]
    return bend_times


def form_bend_array(bend_times, t_p):
    bend_i = 0
    bl = []
    if len(bend_times):
        for i, t in enumerate(t_p):
            if t > bend_times[bend_i]:
                if bend_i < len(bend_times) - 1:
                    bend_i += 1
            bl.append(bend_i)
    return bl


def extract_velocity(tt, xx, yy, settings):
    ftev = settings["frames_to_estimate_velocity"]
    if len(tt) - 1 < ftev:
        raise RuntimeError("Need more frames than frames_to_estimate_velocity")
    dtt = -(np.roll(tt, ftev) - tt)[ftev:]
    dxx = (np.roll(xx, ftev) - xx)[ftev:]
    dyy = (np.roll(yy, ftev) - yy)[ftev:]
    velocity = (settings["px_to_mm"] * settings["fps"]
                * np.median(np.sqrt(dxx**2 + dyy**2) / dtt))
    return velocity


def extract_max_speed(tt, xx, yy, settings):
    ftev = settings["frames_to_estimate_velocity"]
    if len(tt) - 1 < ftev:
        raise RuntimeError("Need more frames than frames_to_estimate_velocity")
    dtt = -(np.roll(tt, ftev) - tt)[ftev:]
    dxx = (np.roll(xx, ftev) - xx)[ftev:]
    dyy = (np.roll(yy, ftev) - yy)[ftev:]
    percentile = (
        settings["px_to_mm"] * settings["fps"] *
        np.percentile((np.sqrt(dxx**2 + dyy**2) / dtt), 90))
    return percentile


def extract_move_per_bend(bl, tt, xx, yy, px_to_mm):
    bend_i = 1
    j = 0
    dists = []
    for i in range(len(bl)):
        if int(bl[i]) == bend_i:
            xi = np.interp(i, tt, xx)
            xj = np.interp(j, tt, xx)
            yi = np.interp(i, tt, yy)
            yj = np.interp(j, tt, yy)

            dist = px_to_mm * np.sqrt((xj - xi)**2 + (yj - yi)**2)
            dists.append(dist)
            bend_i += 1
            j = i

    if len(dists) > 0:
        return np.sum(dists) / len(dists)
    else:
        return np.nan


def write_stats(settings, results, f, paralyzed_stats=True, prepend='',
                mask=None):
    stats = statistics(results, settings, mask)

    f.write(f'\n-------------------------------\n{prepend}\n')

    if settings["cutoff_filter"]:
        if mask is None:
            # Meaningless if mask != None
            f.write('Total particles: %i\n' %
                    results['cutoff_filter_data']['original_particles'])
        else:
            f.write('Total particles: Not saved for regions\n')
    else:
        f.write('Total particles: %i\n' %
                stats['count'])

    if paralyzed_stats and mask is None:
        # filters stats are only meaningful if mask == None
        f.write('\nCUT-OFF tool/filters\n')

        # Not saved for cutoff_filter
        f.write('Max particles present at same time: %i\n'
                % stats['max_number_worms_present'])
        f.write('\n')
        if settings["cutoff_filter"]:
            # Meaningless if mask != None
            f.write('Frame number:       ')
            for item in results['cutoff_filter_data']["frames"]:
                f.write('%i,    ' % item)

            f.write('\n# of particles:   ')
            for item in results['cutoff_filter_data']["list_number"]:
                f.write('%i,    ' % item)

            f.write('\nCut-off tool: Yes\n')
            if settings["use_average"]:
                f.write('Method: averaging\n')
            else:
                f.write('Method: maximum\n')
            f.write(
                'Removed particles: %i\n' %
                results['cutoff_filter_data']['removed_particles_cutoff'])
        else:
            f.write('Cut-off tool: No\n')

        if settings["extra_filter"]:
            f.write('Extra filter: Yes\n')
            f.write(
                'Settings: remove when bpm > %.5f and velocity < %.5f\n' %
                (settings["Bends_max"], settings["Speed_max"]))
            f.write('Removed particles: %i' %
                    results['extra_filter_spurious_worms'])
        else:
            f.write('Extra filter: No\n')

    f.write('\n-------------------------------\n\n')

    f.write(prepend + 'BPM Mean: %.5f\n' % stats['bpm_mean'])
    f.write(prepend + 'BPM Standard deviation: %.5f\n' % stats['bpm_std'])
    f.write(prepend + 'BPM Error on Mean: %.5f\n' % stats['bpm_mean_std'])
    f.write(prepend + 'BPM Median: %.5f\n' % stats['bpm_median'])

    f.write(prepend + 'Bends in movie Mean: %.5f\n' %
            stats['bends_in_movie_mean'])
    f.write(prepend + 'Bends in movie Standard deviation: %.5f\n' %
            stats['bends_in_movie_std'])
    f.write(prepend + 'Bends in movie Error on Mean: %.5f\n' %
            stats['bends_in_movie_mean_std'])
    f.write(
        prepend +
        'Bends in movie Median: %.5f\n' %
        stats['bends_in_movie_median'])

    f.write(prepend + 'Speed Mean: %.6f\n' % stats['vel_mean'])
    f.write(prepend + 'Speed Standard deviation: %.6f\n' % stats['vel_std'])
    f.write(prepend + 'Speed Error on Mean: %.6f\n' % stats['vel_mean_std'])
    f.write(prepend + 'Speed Median: %.6f\n' % stats['vel_median'])

    f.write(
        prepend +
        '90th Percentile speed Mean: %.6f\n' %
        stats['max_speed_mean'])
    f.write(prepend + '90th Percentile speed Standard deviation: %.6f\n' %
            stats['max_speed_std'])
    f.write(prepend + '90th Percentile speed Error on mean: %.6f\n' %
            stats['max_speed_mean_std'])
    if np.isnan(stats['move_per_bend_mean']):
        f.write(prepend + 'Dist per bend Mean: nan\n')
        f.write(prepend + 'Dist per bend Standard deviation: nan\n')
        f.write(prepend + 'Dist per bend Error on Mean: nan\n')
    else:
        f.write(
            prepend +
            'Dist per bend Mean: %.6f\n' %
            stats['move_per_bend_mean'])
        f.write(prepend + 'Dist per bend Standard deviation: %.6f\n' %
                stats['move_per_bend_std'])
        f.write(prepend + 'Dist per bend Error on Mean: %.6f\n' %
                stats['move_per_bend_mean_std'])
    if paralyzed_stats:
        f.write(prepend + 'Moving worms: %i\n' % stats['n_moving'])
        f.write(prepend + 'Paralyzed worms: %i\n' % stats['n_paralyzed'])
        f.write(prepend + 'Total worms: %i\n' %
                stats['max_number_worms_present'])
        f.write(prepend + 'Moving ratio: %.6f\n' %
                (float(stats['n_moving']) / stats['count']))
        f.write(prepend + 'Paralyzed ratio: %.6f\n' %
                (float(stats['n_paralyzed']) / stats['count']))
        if stats['n_paralyzed'] > 0:
            f.write(prepend + 'Moving-to-paralyzed ratio: %.6f\n' % (float(
                stats['n_moving']) / stats['n_paralyzed']))
        else:
            f.write(prepend + 'Moving-to-paralyzed ratio: inf\n')
        if stats['n_moving'] > 0:
            f.write(prepend + 'Paralyzed-to-moving ratio: %.6f\n' % (float(
                stats['n_paralyzed']) / stats['n_moving']))
        else:
            f.write(prepend + 'Paralyzed-to-moving ratio: inf\n')
    f.write(prepend + 'Area Mean: %.6f\n' % stats['area_mean'])
    f.write(prepend + 'Area Standard Deviation: %.6f\n' % stats['area_std'])
    f.write(prepend + 'Area Error on Mean: %.6f\n' % stats['area_mean_std'])

    f.write(prepend + 'Round ratio Mean: %.6f\n' % stats['round_ratio_mean'])
    f.write(prepend + 'Round ratio Standard deviation: %.6f\n' %
            stats['round_ratio_std'])
    f.write(prepend + 'Round ratio Error on mean: %.6f\n' %
            stats['round_ratio_mean_std'])

    f.write(prepend + 'Eccentricity Mean: %.6f\n' % stats['eccentricity_mean'])
    f.write(prepend + 'Eccentricity Standard deviation: %.6f\n' %
            stats['eccentricity_std'])
    f.write(prepend + 'Eccentricity Error on mean: %.6f\n' %
            stats['eccentricity_mean_std'])


def mean_std(x, appears_in):
    mean = np.sum(x * appears_in) / np.sum(appears_in)
    second_moment = np.sum(x**2 * appears_in) / np.sum(appears_in)
    std = np.sqrt(second_moment - mean**2)
    return mean, std


def statistics(results, settings, mask=None):
    df = results["particle_dataframe"]
    if mask is None:
        mask = df.loc[:, "Appears in frames"] > 0

    df = df.loc[mask, :]

    P = results["track"]['particle']
    T = results["track"]['frame']

    if settings["cutoff_filter"]:
        max_number_worms_present = len(df)
    else:
        max_number_worms_present = max(
            [len([1 for p in set(P[T == t]) if p in df.index])
             for t in set(T)])
    count = len(df)
    n_moving = np.sum(df.loc[:, "Moving"])
    n_paralyzed = len(df) - n_moving

    appears_in = df.loc[:, "Appears in frames"]
    bpm_mean, bpm_std = mean_std(df.loc[:, "BPM"], appears_in)
    bpm_median = np.median(df.loc[:, "BPM"])
    bpm_mean_std = bpm_std / np.sqrt(max_number_worms_present)

    bends_in_movie_mean, bends_in_movie_std = mean_std(
        df.loc[:, "bends_in_movie"], appears_in)
    bends_in_movie_median = np.median(df.loc[:, "bends_in_movie"])
    bends_in_movie_mean_std = bends_in_movie_std / \
        np.sqrt(max_number_worms_present)

    vel_mean, vel_std = mean_std(df.loc[:, "Speed"], appears_in)
    vel_mean_std = vel_std / np.sqrt(max_number_worms_present)
    vel_median = np.median(df.loc[:, "Speed"])

    area_mean, area_std = mean_std(df.loc[:, "Area"], appears_in)
    area_mean_std = area_std / np.sqrt(max_number_worms_present)

    max_speed_mean, max_speed_std = mean_std(
        df.loc[:, "Max speed"], appears_in)
    max_speed_mean_std = max_speed_std / np.sqrt(max_number_worms_present)

    round_ratio_mean, round_ratio_std = mean_std(
        df.loc[:, "Round ratio"], appears_in)
    round_ratio_mean_std = round_ratio_std / np.sqrt(max_number_worms_present)

    eccentricity_mean, eccentricity_std = mean_std(
        df.loc[:, "eccentricity"], appears_in)
    eccentricity_mean_std = eccentricity_std / \
        np.sqrt(max_number_worms_present)

    # Ignore nan particles for move_per_bend
    mask_appear = np.logical_not(np.isnan(df.loc[:, "Dist per bend"]))
    if np.any(mask_appear):
        move_per_bend_mean, move_per_bend_std = mean_std(
            df.loc[mask_appear, "Dist per bend"],
            df.loc[mask_appear, "Appears in frames"])
        move_per_bend_mean_std = move_per_bend_std / \
            np.sqrt(max([np.sum(mask_appear), max_number_worms_present]))
    else:
        move_per_bend_mean = np.nan
        move_per_bend_std = np.nan
        move_per_bend_mean_std = np.nan

    stats = {
        'max_number_worms_present': max_number_worms_present,
        'n_paralyzed': n_paralyzed,
        'n_moving': n_moving,
        'bpm_mean': bpm_mean,
        'bpm_std': bpm_std,
        'bpm_median': bpm_median,
        'bpm_mean_std': bpm_mean_std,
        'bends_in_movie_mean': bends_in_movie_mean,
        'bends_in_movie_std': bends_in_movie_std,
        'bends_in_movie_mean_std': bends_in_movie_mean_std,
        'bends_in_movie_median': bends_in_movie_median,
        'vel_mean': vel_mean,
        'vel_std': vel_std,
        'vel_mean_std': vel_mean_std,
        'vel_median': vel_median,
        'area_mean': area_mean,
        'area_std': area_std,
        'area_mean_std': area_mean_std,
        'max_speed_mean': max_speed_mean,
        'max_speed_std': max_speed_std,
        'max_speed_mean_std': max_speed_mean_std,
        'move_per_bend_mean': move_per_bend_mean,
        'move_per_bend_std': move_per_bend_std,
        'move_per_bend_mean_std': move_per_bend_mean_std,
        'count': count,
        'round_ratio_mean': round_ratio_mean,
        'round_ratio_std': round_ratio_std,
        'round_ratio_mean_std': round_ratio_mean_std,
        'eccentricity_mean': eccentricity_mean,
        'eccentricity_std': eccentricity_std,
        'eccentricity_mean_std': eccentricity_mean_std}

    return stats


def write_particles(settings, particles_dataframe, filename):
    """Write particles dataframe to csv"""
    df = particles_dataframe.loc[:, [
        "BPM", "bends_in_movie", "Speed", "Max speed", "Dist per bend",
        "Area", "Appears in frames", "Moving", "Region", "Round ratio",
        "eccentricity"]]

    x = (settings["limit_images_to"] / settings["fps"])
    df.columns = [
        'BPM', f'Bends per {x:.2f} s', 'Speed', 'Max speed', 'Dist per bend',
        'Area', 'Appears in frames', 'Moving (non-paralyzed)', 'Region',
        'Round ratio', 'Eccentricity']

    df.to_csv(filename)


def write_results_file(results, settings):
    df = results["particle_dataframe"]
    write_particles(settings,
                    df,
                    os.path.join(settings["save_as"], 'particles.csv'))

    with open(os.path.join(settings["save_as"], 'results.txt'), 'w') as f:
        f.write('---------------------------------\n')
        f.write('    Results for %s \n' % settings["video_filename"])
        f.write('---------------------------------\n\n')

        # Stats for all worms
        write_stats(settings, results, f, paralyzed_stats=True)

        # Stats for moving worms
        moving_mask = df.loc[:, "Moving"]

        write_stats(settings, results, f, paralyzed_stats=False,
                    prepend='Moving ', mask=moving_mask)

        # Raw stats
        f.write('---------------------------------\n\n')

        regions = settings["regions"]
        # Per region stats
        if len(regions) > 1:
            for reg in regions:
                f.write('---------------------------------\n')
                f.write('Stats for region: %s\n' % reg)
                f.write('---------------------------------\n\n')

                # Worms of this region
                try:
                    pars = np.asarray(results["region_particles"][reg], int)
                except TypeError:
                    pars = [int(results["region_particles"][reg])]
                if len(pars) == 0:
                    f.write('Nothing found in region.\n\n')
                    continue
                indices = [idx for idx in pars if idx in df.index]

                # All worms
                write_stats(settings, results, f, paralyzed_stats=True,
                            mask=indices)

                f.write('\n\n')
        f.write('\n')

    print('results.txt file produced.')


# =============================================================================
# --- Matplotlib code---
# =============================================================================
def print_frame(settings, t, P, T, bends, track):
    font = {'size': settings["font_size"]}
    print('Printing frame', t + 1)
    image_filename = os.path.join(
        settings["save_as"], 'imgs', '%05d.jpg' % (int(t)))
    frame = (255 - io.imread(image_filename))
    os.remove(image_filename)
    small_imshow(settings, frame, cmap=cm.binary, vmax=300)
    for p in bends.index:
        pp = P == p
        l = np.logical_and(pp, T == t)
        if np.sum(l) > 0:
            x = track['x'][l].iloc[0]
            y = track['y'][l].iloc[0]
            b = bends[p][np.sum(T[pp] < t)]
            plt.text(y + 3, x + 3, 'p=%i\n%.1f' %
                     (p, b), font, color=[1, 0.3, 0.2])

    m, n = frame.shape
    plt.plot(
        [n - (5 + settings["scale_bar_size"] / float(settings["px_to_mm"])),
         n - 5],
        [m - 5, m - 5],
        linewidth=settings["scale_bar_thickness"], c=[0.5, 0.5, 0.5])
    plt.axis('off')
    plt.axis('tight')
    plt.savefig(os.path.join(settings["save_as"], 'imgs', '%05d.jpg' % (t)))


def print_images(settings, bends):
    plt.gcf().set_size_inches(20, 20)
    plt.clf()
    with open(os.path.join(settings["save_as"], 'track.p'),
              'br') as trackfile:
        track = pickle.load(trackfile)
    P = track['particle']
    T = track['frame']
    output_overlayed_images = settings["output_overlayed_images"]
    if output_overlayed_images != 0:
        up_to = (len(set(T)) if output_overlayed_images is None
                 else output_overlayed_images)
        for t in range(up_to):
            print_frame(settings, t, P, T, bends, track)
    plt.clf()


def small_imshow(settings, img, *args, **kwargs):
    # For large images/frames matplotlib's imshow gives memoryerror
    # This is solved by resizing before plotting
    max_pixels = settings["max_plot_pixels"]
    original_shape = img.shape
    if np.product(img.shape) > max_pixels:
        factor = max_pixels / np.product(img.shape)
        img = resize(
            np.asarray(img, float),
            (int(img.shape[0] * factor), int(img.shape[1] * factor)),
            preserve_range=True)
    plt.clf()
    plt.imshow(img, *args, extent=[0, original_shape[1],
                                   original_shape[0], 0], **kwargs)


def output_processing_frames(
        settings, save_folder, frameorig, Z, frame, thresholded,
        frame_after_open, frame_after_close, labeled,
        labeled_removed, skel_labeled=None):

    plt.gcf().set_size_inches(20, 20)
    plt.clf()
    small_imshow(settings, frameorig, cmap=cm.gray)
    plt.savefig(os.path.join(save_folder, '0frameorig.jpg'))

    small_imshow(settings, Z, cmap=cm.gray)
    plt.savefig(os.path.join(save_folder, '0z.jpg'))

    small_imshow(settings, frame, cmap=cm.gray)
    plt.savefig(os.path.join(save_folder, '1framesubtract.jpg'))

    small_imshow(settings, thresholded, cmap=cm.binary)
    plt.savefig(os.path.join(save_folder, '2thresholded.jpg'))

    small_imshow(settings, frame_after_open, cmap=cm.binary)
    plt.savefig(os.path.join(save_folder, '3opened.jpg'))

    small_imshow(settings, frame_after_close, cmap=cm.binary)
    plt.savefig(os.path.join(save_folder, '4closed.jpg'))

    small_imshow(settings, labeled, cmap=cm.binary)
    plt.savefig(os.path.join(save_folder, '5labelled.jpg'))

    small_imshow(settings, labeled_removed, cmap=cm.binary)
    plt.savefig(os.path.join(save_folder, '6removed.jpg'))

    if skel_labeled is not None:
        small_imshow(settings, skel_labeled, cmap=cm.binary)
        plt.savefig(os.path.join(save_folder, '7skeletonized.jpg'))
    plt.clf()


def print_example_frame(
        settings, sizes, save_folder, frameorig, Z, frame, thresholded,
        frame_after_open, frame_after_close, labeled, labeled_removed,
        skel_labeled):
    print('Sizes:')
    print(sizes)

    output_processing_frames(
        settings, save_folder, frameorig, Z, frame, thresholded,
        frame_after_open, frame_after_close, labeled, labeled_removed,
        (skel_labeled if settings["skeletonize"] else None))
    print('Example frame outputted!')
