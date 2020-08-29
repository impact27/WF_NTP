#!/usr/bin/env python3
"""
Copyright (C) 2019 Quentin Peter

This file is part of WF_NTP.

WF_NTP is distributed under CC BY-NC-SA version 4.0. You should have
recieved a copy of the licence along with WF_NTP. If not, see
https://creativecommons.org/licenses/by-nc-sa/4.0/.
"""
import matplotlib.pyplot as plt
from functools import partial
from copy import deepcopy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
import tkinter.messagebox
import tkinter.simpledialog
import cv2
import os
import matplotlib
import traceback
import json
import sys
import faulthandler
import multiprocessing as mp
from queue import Empty

from WF_NTP.defaults import DEFAULTS
from WF_NTP.plot_path import plot_path
from WF_NTP.WF_NTP_script import (
    run_tracker as _run_tracker)
from WF_NTP.WF_NTP_script import (
    print_images, print_example_frame, StdoutRedirector, save_settings)


class MainApplication(tk.Frame):

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.jobindex = 0
        self.jobs = {}
        self.job_buttons = {}
        self.pool = mp.Pool()
        self.manager = mp.Manager()
        self.stdout_queue = self.manager.Queue()
        self.editing_job = False
        self.parent.after(100, self.update_stdout)
        sys.stdout = StdoutRedirector(self.stdout_queue)

        # Main windo
        parent.wm_title("Wide-field nematode tracking platform")

        # Overview
        frame_job_list = tk.Frame(self, relief=tk.GROOVE, bd=1)
        canvas = tk.Canvas(frame_job_list)
        logscrollbar = tk.Scrollbar(frame_job_list, orient="vertical",
                                    command=canvas.yview)
        self.job_list_container = tk.Frame(canvas)

        # Rescale scrollbar
        def rescale_logscrollbar(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        self.job_list_container.bind("<Configure>", rescale_logscrollbar)

        # Add job container in canvas
        canvas_frame = canvas.create_window(
            (0, 0), window=self.job_list_container, anchor='nw')
        canvas.configure(yscrollcommand=logscrollbar.set)

        def resize_container(event):
            canvas_width = event.width
            canvas.itemconfig(canvas_frame, width=canvas_width)

        canvas.bind("<Configure>", resize_container)
        # Pack everything
        canvas.pack(side='left', fill=tk.BOTH, expand=True)
        logscrollbar.pack(side="right", fill="y")
        frame_job_list.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # =====================================================================
        #       # Console
        # =====================================================================
        frame_console = tk.Frame(self, relief=tk.GROOVE, bd=1)

        self.console = tk.Text(frame_console, state=tk.DISABLED)
        scroll = tk.Scrollbar(frame_console)
        self.console.configure(yscrollcommand=scroll.set)

        self.console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        frame_console.pack(side='top', fill=tk.BOTH, expand=True)

        # =====================================================================
        #       # Buttons
        # =====================================================================
        frame_buttons = tk.Frame(self)

        start_btn = tk.Button(
            frame_buttons, text="Start", command=self.start_all)
        addjob_btn = tk.Button(
            frame_buttons, text="Add job", command=self.add_job_dialog)
        load_btn = tk.Button(
            frame_buttons, text="Load job", command=self.load_job)
        utilities_btn = tk.Button(
            frame_buttons, text="Utilities", command=self.utils)

        start_btn.grid(row=0, column=0)
        addjob_btn.grid(row=0, column=1)
        load_btn.grid(row=0, column=2)
        utilities_btn.grid(row=0, column=3)
        frame_buttons.pack(side='bottom', fill=tk.X, expand=False)

        def on_closing():
            self.pool.terminate()
            sys.stdout = sys.__stdout__
            self.parent.destroy()
            self.parent.quit()

        self.parent.protocol("WM_DELETE_WINDOW", on_closing)

    def update_stdout(self):
        empty = False
        while not empty:
            try:
                self.log(self.stdout_queue.get_nowait())
            except Empty:
                empty = True
        self.parent.after(100, self.update_stdout)

    def utils(self):
        Utils(self)

    def start_all(self):
        for key in self.jobs:
            self.start_job(key)

    def start_job(self, index, example=False):
        job = self.jobs[index]
        job["stop_after_example_output"] = example
        job_buttons = self.job_buttons[index]
        job_buttons['progressbar'].start()

        self.run_tracker(job, index, example)

    def run_tracker(self, settings, index, example=False):

        def finish(args):
            if not args:
                return
            settings, print_data, bends = args
            self.parent.after(
                1, self.finished, settings, index, print_data,
                bends, example)

        def error_callback(error):
            self.stdout_queue.put(''.join(traceback.format_exception_only(
                error.__class__, error)))

        settings["stdout prefix"] = f"[{index}]"
        self.pool.apply_async(run_tracker, (settings, self.stdout_queue),
                              callback=finish, error_callback=error_callback)

    def finished(self, settings, index, print_data, bends, example=False):
        print_example_frame(settings, *print_data)
        if bends is not None:
            print_images(settings, bends)
        if example:
            self.log('Finished example output ' + settings['video_filename'])
        else:
            self.log('Finished ' + settings['video_filename'])
        self.job_buttons[index]['progressbar'].stop()

    def log(self, txt):
        # Print it in the terminal as well
        print(txt, file=sys.__stdout__)
        sys.__stdout__.flush()
        self.console.config(state=tk.NORMAL)
        if txt[-1] != '\n':
            txt += '\n'
        self.console.insert(tk.END, txt)
        self.console.see('end')
        self.console.config(state=tk.DISABLED)

    def add_job_dialog(self):
        AddJob(self)

    def add_job(self, job):
        # Fill default values
        for key in DEFAULTS:
            if key not in job:
                job[key] = DEFAULTS[key][0]
        videonames = job['video_filename'].split(', ')
        for videoname in videonames:
            i = self.jobindex
            this_job = deepcopy(job)
            this_job['video_filename'] = videoname

            short_videoname = videoname.split('/')[-1]
            append_name = ".".join(short_videoname.split('.')[:-1])

            if len(videonames) > 1:
                this_job['save_as'] += '_' + append_name

            self.jobs[i] = this_job

            if len(short_videoname) > 25:
                short_videoname = short_videoname[:25]
            short_outputname = this_job['save_as']

            # TK CODE

            thisframe = tk.Frame(self.job_list_container)

            deletebtn = tk.Button(
                thisframe, text='X',
                command=partial(self.delete_job, i))
            videobtn = tk.Button(
                thisframe, text=short_videoname,
                command=partial(self.edit_job, i))
            examplebtn = tk.Button(
                thisframe, text="Example",
                command=partial(self.start_job, i, True))
            jobinfo = tk.Label(thisframe, text=short_outputname, width=10)

            progressbar = ttk.Progressbar(
                thisframe, orient='horizontal',
                length=140, mode='indeterminate')

            deletebtn.pack(side='left', fill=tk.X)
            videobtn.pack(side='left', fill=tk.X)
            examplebtn.pack(side='left', fill=tk.X)
            jobinfo.pack(side='left', fill=tk.X, expand=True)
            progressbar.pack(side='right', fill=tk.X)
            thisframe.pack(side='top', fill=tk.X)

            self.job_buttons[i] = {'videobtn': videobtn,
                                   'jobinfo': jobinfo,
                                   'progressbar': progressbar,
                                   'deletebtn': deletebtn,
                                   'thisframe': thisframe}
            if not self.editing_job:
                self.log('Job: "%s" successfully added.'
                         % videoname.split('/')[-1])
            self.jobindex += 1
            self.editing_job = False
            # Save settings
            save_settings(self.jobs[i])

    def edit_job(self, index):
        self.editing_job = True
        self.editing_index = index
        AddJob(self)

    def load_job(self):
        filenames = filenames = tkinter.filedialog.askopenfilenames(
            title='Locate a settings.json file', filetypes=[
                ("Settings file", "*.json")])
        if not filenames:
            return
        for filename in filenames:
            try:
                with open(filename) as f:
                    job = json.load(f)
            except Exception:
                self.log('Not a valid settings.json file')
                self.log("".join(traceback.format_exc()))
                return
            try:
                path_keys = ['video_filename', 'save_as']
                for key in path_keys:
                    if os.path.isabs(job[key]):
                        continue
                    folder = os.path.dirname(filename)
                    new_path = os.path.join(folder, job[key])
                    job[key] = os.path.abspath(new_path)

            except Exception:
                self.log('Not a valid settings.json file')
                self.log("".join(traceback.format_exc()))
                return
            self.add_job(job)

    def delete_job(self, index):
        name = self.jobs[index]['video_filename'].split('/')[-1]
        if tkinter.messagebox.askokcancel(
                "Delete " + name, "Are you sure you wish delete this job?"):
            self.job_buttons[index]['thisframe'].pack_forget()
            self.log('Deleted job "%s".' % name)
            del self.job_buttons[index]
            del self.jobs[index]


def run_tracker(settings, queue=None):
    try:
        if settings["stop_after_example_output"]:
            queue.put('Job: "{}" example output started.'.format(
                settings['video_filename'].split('/')[-1]))
        else:
            queue.put('Job: "{}" started.'.format(
                settings['video_filename'].split('/')[-1]))
        print_data, bends = _run_tracker(settings, queue)
        return settings, print_data, bends
    except Exception:
        queue.put(
            "Got error while processing, please check the parameters.")
        queue.put("The error was:")
        # Print the full traceback to the terminal
        queue.put(''.join(traceback.format_exc()))


class Utils(tk.Toplevel):
    def __init__(self, parent, *args, **kwargs):
        tk.Toplevel.__init__(self, *args, **kwargs)
        self.wm_title("Utilities")
        self.grab_set()
        self.parent = parent

        tk.Button(self, text="Plot path", width=30,
                  command=self.plotpath).pack(fill=tk.X)

        tk.Button(self, text="Export to tsv", width=30,
                  command=self.tsv).pack(fill=tk.X)

    def plotpath(self):
        filename = tkinter.filedialog.askopenfilename(
            title='Locate a track.p file', filetypes=[("track.p file", "*.p")])
        if filename:
            plot_path(filename)

    def tsv(self):
        ToTsv(self)


class ToTsv(tk.Toplevel):
    def __init__(self, parent, *args, **kwargs):
        tk.Toplevel.__init__(self, *args, **kwargs)
        self.parent = parent
        self.main = parent.parent
        # Main window
        self.wm_title("Export tab-seperated file")
        self.grab_set()

        self.index = 0
        self.filenames = []

        # Overview
        logframe = tk.Frame(self, relief=tk.GROOVE, bd=1)
        logframe.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(logframe)
        self.logframe = tk.Frame(canvas)

        logscrollbar = tk.Scrollbar(logframe, orient="vertical",
                                    command=canvas.yview)
        canvas.configure(yscrollcommand=logscrollbar.set)

        logscrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill=tk.BOTH, expand=True)
        canvas.create_window((0, 0), window=self.logframe, anchor='nw')

        def rescale_logscrollbar(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        self.logframe.bind("<Configure>", rescale_logscrollbar)
        canvas.configure(scrollregion=canvas.bbox("all"))

        btnframe = tk.Frame(self, bd=1)
        btnframe.pack(fill=tk.X, expand=False)

        tk.Button(
            btnframe, text='Add files', command=self.add).grid(
                row=0, column=0)
        tk.Button(
            btnframe, text='Add folder', command=self.addrecursive).grid(
                row=0, column=1)
        tk.Button(
            btnframe, text='Export', command=self.export).grid(
                row=0, column=2)
        tk.Button(
            btnframe, text="Close", command=self.destroy).grid(
                row=0, column=3)

    def add_filesnames(self, filenames):
        for f in filenames:
            self.filenames.append(f)
            thisframe = tk.Frame(self.logframe, width=300, height=300)
            thisframe.grid(row=self.index, column=0, sticky='w')
            tk.Label(thisframe, text=f).pack()
            self.index += 1

    def add(self):
        filenames = tkinter.filedialog.askopenfilenames(
            title='Locate results.txt files', filetypes=[
                ("results.txt files", "*.txt")])
        self.add_filesnames(filenames)

    def addrecursive(self):
        folder = tkinter.filedialog.askdirectory()
        if folder and tkinter.messagebox.askquestion(
                "Warning",
                "Recursive searching might take a while for large folders.\n"
                "Are you sure you wish to search this folder?",
                icon='warning'):
            filenames = []
            for f in os.walk(folder):
                if 'results.txt' in f[2]:
                    filenames.append(
                        (f[0].rstrip('/\\') + '/results.txt').replace(
                            '\\', '/'))
            self.add_filesnames(filenames)

    def export(self):
        fnames = self.filenames

        output = []
        first = True
        legends = ['Saved as', 'Movie', 'Region']

        for fname in fnames:
            sep = '---------------------------------'

            save_as = "/".join(fname.split('/')[:-1])
            with open(fname) as f:
                s = f.read()
            if sep not in s:
                self.main.log(fname + ' not a results.txt file.')
                continue

            regions = 'Stats for region:'
            skip_first = regions in s

            l = s.split(sep)
            parse_next = False
            moviename, region_name = '', ''
            for section in l:
                if parse_next:
                    pars = [save_as, moviename, region_name]
                    lines = section.split('\n')
                    for line in lines:
                        if ':' in line:
                            s, n = [x.strip() for x in line.split(':')]
                            if first:
                                legends.append(s)
                            pars.append(n)
                    output.append(pars)

                    parse_next = False
                    first = len(legends) == 3

                elif 'Results for' in section:
                    parse_next = not skip_first
                    moviename = section.split('Results for')[-1].strip()
                    region_name = 'all'

                elif 'Stats for region:' in section:
                    parse_next = True
                    region_name = section.split(
                        'Stats for region:')[-1].strip()

        if len(output) > 0:
            out = ''
            for i in range(len(legends)):
                out += legends[i]
                for j in range(len(output)):
                    if len(output[j]) > i:
                        out += '\t' + output[j][i]
                    elif len(output[j]) == i:
                        out += '\t' + '0'
                    else:
                        out += '\t' + 'n/a'
                out += '\n'

            save_fname = tkinter.filedialog.asksaveasfilename(
                filetypes=[('*.tsv', 'Tab seperated file')])
            if save_fname:
                if save_fname[-4:] != '.tsv':
                    save_fname += '.tsv'
                with open(save_fname, 'w') as f:
                    f.write(out)

                self.main.log(save_fname + ' written.')


class AddJob(tk.Toplevel):
    def __init__(self, parent, *args, **kwargs):
        tk.Toplevel.__init__(self, *args, **kwargs)
        self.wm_title("Add job")
        self.grab_set()
        self.parent = parent
        padframes = 7
        self.regions = {}
        self.editing = parent.editing_job
        self.edit_job = None
        if self.editing:
            self.edit_index = parent.editing_index
            self.edit_job = parent.jobs[self.edit_index]
        editing = self.editing
        self.resizable(False, False)

        self.adding_roi = False
        self.widgets = {}

        # ----- VIDEO SECTION -----

        videoframe = tk.LabelFrame(self, text="Video")

        videonameframe = tk.Frame(videoframe)
        videoname = tk.Entry(videonameframe, width=80, state='readonly')
        video_btn = tk.Button(videonameframe, text="Browse",
                              command=self.find_video)
        self.videoinfo = tk.Label(
            videonameframe, text='No video chosen.')

        videocutframe = tk.Frame(videoframe)
        start_frame_label = tk.LabelFrame(videocutframe, text="Start frame")
        start_frame = tk.Entry(start_frame_label, width=15)
        use_frame_label = tk.LabelFrame(videocutframe, text="Use frames")
        use_frame = tk.Entry(use_frame_label, width=15)
        fps_label = tk.LabelFrame(videocutframe, text="FPS")
        fps = tk.Entry(fps_label, width=15)
        px_to_mm_label = tk.LabelFrame(videocutframe, text="px to mm factor")
        px_to_mm = tk.Entry(px_to_mm_label, width=15)
        darkfield_label = tk.LabelFrame(videocutframe, text="Darkfield")
        darkfield = tk.IntVar()
        darkfield_box = tk.Checkbutton(darkfield_label, variable=darkfield)

        self.widgets['video_filename'] = videoname
        self.widgets['start_frame'] = start_frame
        self.widgets['limit_images_to'] = use_frame
        self.widgets['fps'] = fps
        self.widgets['px_to_mm'] = px_to_mm
        self.widgets['darkfield'] = darkfield

        # Pack and grid
        videoframe.grid(row=0, sticky='w')
        videonameframe.grid(row=0)
        videocutframe.grid(row=1, sticky='w')

        videoname.grid(row=0, padx=10)
        video_btn.grid(row=0, column=1, padx=5)
        self.videoinfo.grid(row=1, sticky='w', padx=10)

        start_frame_label.grid(row=2, column=0, sticky='w', padx=10)
        use_frame_label.grid(row=2, column=1, sticky='w', padx=10)
        fps_label.grid(row=2, column=2, sticky='w', padx=10)
        px_to_mm_label.grid(row=2, column=3, sticky='w', padx=10)
        darkfield_label.grid(row=2, column=4, sticky='w', padx=10)

        start_frame.pack(side='left', padx=3)
        use_frame.pack(side='left', padx=3)
        fps.pack(side='left', padx=3)
        px_to_mm.pack(side='left', padx=3)
        darkfield_box.pack()

        # ----- LOCATING SECTION -----

        locate_frame = tk.LabelFrame(self, text="Locating")
        methodframe = tk.Frame(locate_frame)
        method_label = tk.LabelFrame(methodframe, text="Method")
        method = ttk.Combobox(method_label, values=(
            'Z-filtering', 'Keep Paralyzed'), state='readonly')
        std_px_label = tk.LabelFrame(methodframe, text="Std pixels")
        std_px = tk.Entry(std_px_label, width=15)
        threshold_label = tk.LabelFrame(
            methodframe, text="Threshold (0-255)")
        threshold = tk.Entry(threshold_label, width=15)
        opening_label = tk.LabelFrame(methodframe, text="Opening")
        opening = tk.Entry(opening_label, width=15)
        closing_label = tk.LabelFrame(methodframe, text="Closing")
        closing = tk.Entry(closing_label, width=15)
        skeletonize_label = tk.LabelFrame(
            methodframe, text="Skeletonize")
        skeletonize = tk.IntVar()
        skeletonize_box = tk.Checkbutton(
            skeletonize_label, variable=skeletonize)
        prune_size_label = tk.LabelFrame(
            methodframe, text="Prune size")
        prune_size = tk.Entry(prune_size_label, width=15)
        do_full_prune_label = tk.LabelFrame(
            methodframe, text="Full prune")
        do_full_prune = tk.IntVar()
        do_full_prune_box = tk.Checkbutton(
            do_full_prune_label, variable=do_full_prune)

        self.widgets['keep_paralyzed_method'] = method
        self.widgets['std_px'] = std_px
        self.widgets['threshold'] = threshold
        self.widgets['opening'] = opening
        self.widgets['closing'] = closing
        self.widgets['prune_size'] = prune_size
        self.widgets['skeletonize'] = skeletonize
        self.widgets['do_full_prune'] = do_full_prune

        locate_frame.grid(row=1, sticky='w', pady=padframes)
        methodframe.grid(row=0)

        method_label.grid(row=0, column=0, sticky='w', padx=10)
        std_px_label.grid(row=0, column=1, sticky='w', padx=10)
        opening_label.grid(row=0, column=2, sticky='w', padx=10)
        closing_label.grid(row=0, column=3, sticky='w', padx=10)
        threshold_label.grid(row=1, column=0, sticky='w', padx=10)
        skeletonize_label.grid(row=1, column=1, sticky='w', padx=10)
        prune_size_label.grid(row=1, column=2, sticky='w', padx=10)
        do_full_prune_label.grid(row=1, column=3, sticky='w', padx=10)

        method.pack(side='left', padx=3)
        std_px.pack(side='left', padx=3)
        threshold.pack(side='left', padx=3)
        opening.pack(side='left', padx=3)
        closing.pack(side='left', padx=3)
        skeletonize_box.pack()
        prune_size.pack(side='left', padx=3)
        do_full_prune_box.pack()

        # ----- FILTER  SECTION -----

        filter_frame = tk.LabelFrame(self, text="Filtering")
        minsize_label = tk.LabelFrame(
            filter_frame, text="Minimum size (px)")
        maxsize_label = tk.LabelFrame(
            filter_frame, text="Maximum size (px)")
        minsize = tk.Entry(minsize_label, width=15)
        maxsize = tk.Entry(maxsize_label, width=15)
        minimum_ecc_label = tk.LabelFrame(
            filter_frame, text="Worm-like (0-1)")
        minimum_ecc = tk.Entry(minimum_ecc_label, width=15)

        self.widgets['min_size'] = minsize
        self.widgets['max_size'] = maxsize
        self.widgets['minimum_ecc'] = minimum_ecc

        filter_frame.grid(row=2, sticky='w')
        minsize_label.grid(row=0, column=0, sticky='w', padx=10)
        minsize.pack(side='left', padx=3)
        maxsize_label.grid(row=0, column=1, sticky='w', padx=10)
        maxsize.pack(side='left', padx=3)
        minimum_ecc_label.grid(row=0, column=2, sticky='w', padx=10)
        minimum_ecc.pack(side='left', padx=3)

        # ----- CUT-OFF SECTION -----

        cut_off_frame = tk.LabelFrame(
            self, text=(
                "\n Cut-off tools(choose frames in which "
                "the number of particles is set as cut-off) and extra filter"))
        cutoff_filter_label = tk.LabelFrame(
            cut_off_frame, text="Cut-off")
        cutoff_filter = tk.IntVar()
        cutoff_filter_box = tk.Checkbutton(
            cutoff_filter_label, variable=cutoff_filter)
        use_average_label = tk.LabelFrame(
            cut_off_frame, text="Average or Max")
        use_average = ttk.Combobox(use_average_label, values=(
            'Maximum', 'Average'), state='readonly', width=8)
        lower_label = tk.LabelFrame(
            cut_off_frame, text="Start frame")
        lower = tk.Entry(lower_label, width=7)
        upper_label = tk.LabelFrame(
            cut_off_frame, text="End frame")
        upper = tk.Entry(upper_label, width=7)
        extra_filter_label = tk.LabelFrame(
            cut_off_frame, text="Extra filter")
        extra_filter = tk.IntVar()
        extra_filter_box = tk.Checkbutton(
            extra_filter_label, variable=extra_filter)
        Bends_max_label = tk.LabelFrame(
            cut_off_frame, text="Max Bends")
        Bends_max = tk.Entry(Bends_max_label, width=7)
        Speed_max_label = tk.LabelFrame(
            cut_off_frame, text="Max Speed")
        Speed_max = tk.Entry(Speed_max_label, width=7)

        self.widgets['use_average'] = use_average
        self.widgets['lower'] = lower
        self.widgets['upper'] = upper
        self.widgets['Bends_max'] = Bends_max
        self.widgets['Speed_max'] = Speed_max
        self.widgets['extra_filter'] = extra_filter
        self.widgets['cutoff_filter'] = cutoff_filter

        cut_off_frame.grid(row=3, sticky='w')
        cutoff_filter_label.grid(row=0, column=0, sticky='w', padx=10)
        cutoff_filter_box.pack()
        use_average_label.grid(row=0, column=1, sticky='w', padx=10)
        use_average.pack(side='left', padx=3)
        lower_label.grid(row=0, column=2, sticky='w', padx=5)
        lower.pack(side='left', padx=3)
        upper_label.grid(row=0, column=3, sticky='w', padx=5)
        upper.pack(side='left', padx=3)
        extra_filter_label.grid(row=0, column=4, sticky='w', padx=10)
        extra_filter_box.pack()
        Bends_max_label.grid(row=0, column=5, sticky='w', padx=5)
        Bends_max.pack(side='left', padx=3)
        Speed_max_label.grid(row=0, column=6, sticky='w', padx=5)
        Speed_max.pack(side='left', padx=3)

        # ------ TRAJECTORIES  SECTION ------

        trajs_frame = tk.LabelFrame(self, text="Forming trajectories")
        maxdist_label = tk.LabelFrame(
            trajs_frame, text="Maximum move distance (px)")
        maxdist = tk.Entry(maxdist_label, width=15)
        minlength_label = tk.LabelFrame(
            trajs_frame, text="Minimum length (frames)")
        minlength = tk.Entry(minlength_label, width=15)
        memory_label = tk.LabelFrame(
            trajs_frame, text="Memory (frames)")
        memory = tk.Entry(memory_label, width=15)

        self.widgets['max_dist_move'] = maxdist
        self.widgets['min_track_length'] = minlength
        self.widgets['memory'] = memory

        trajs_frame.grid(row=4, sticky='w', pady=padframes)
        maxdist_label.grid(row=0, column=0, sticky='w', padx=10)
        maxdist.pack(side='left', padx=3)
        minlength_label.grid(row=0, column=1, sticky='w', padx=10)
        minlength.pack(side='left', padx=3)
        memory_label.grid(row=0, column=2, sticky='w', padx=10)
        memory.pack(side='left', padx=3)

        # ------ BENDS/VELOCITY  SECTION ------

        benvel_frame = tk.LabelFrame(self, text="Bends and Velocity")
        bendthres_label = tk.LabelFrame(
            benvel_frame, text="Bend threshold")
        bendthres = tk.Entry(bendthres_label, width=15)
        minbends_label = tk.LabelFrame(
            benvel_frame, text="Minimum bends")
        minbends = tk.Entry(minbends_label, width=15)
        velframes_label = tk.LabelFrame(
            benvel_frame, text="Frames to estimate velocity")
        velframes = tk.Entry(velframes_label, width=15)

        self.widgets['bend_threshold'] = bendthres
        self.widgets['minimum_bends'] = minbends
        self.widgets['frames_to_estimate_velocity'] = velframes

        benvel_frame.grid(row=5, sticky='w')
        bendthres_label.grid(row=0, column=0, sticky='w', padx=10)
        bendthres.pack(side='left', padx=3)
        minbends_label.grid(row=0, column=1, sticky='w', padx=10)
        minbends.pack(side='left', padx=3)
        velframes_label.grid(row=0, column=2, sticky='w', padx=10)
        velframes.pack(side='left', padx=3)

        # ------ TRAJECTORIES  SECTION ------

        paralyzed_frame = tk.LabelFrame(
            self, text="Paralyzed worm statistics")
        maxbpm_label = tk.LabelFrame(
            paralyzed_frame, text="Maximum bends per minute")
        maxbpm = tk.Entry(maxbpm_label, width=15)
        maxvel_label = tk.LabelFrame(
            paralyzed_frame, text="Maximum velocity (mm/s)")
        maxvel = tk.Entry(maxvel_label, width=15)

        self.widgets['maximum_bpm'] = maxbpm
        self.widgets['maximum_velocity'] = maxvel

        paralyzed_frame.grid(row=6, sticky='w', pady=10)
        maxbpm_label.grid(row=0, column=0, sticky='w', padx=10)
        maxbpm.pack(side='left', padx=3)
        maxvel_label.grid(row=0, column=1, sticky='w', padx=10)
        maxvel.pack(side='left', padx=3)

        # ------ REGION OF INTERESTS SECTION ------

        roi_frame = tk.LabelFrame(self, text="Region of interests")
        roi_btn_add = tk.Button(
            roi_frame, text="Add new", command=self.add_roi)
        rois = ttk.Combobox(roi_frame, state='disabled')
        self.roi_btn_show = tk.Button(
            roi_frame, text="Show", command=self.show_roi, state='disabled')
        self.roi_btn_edit = tk.Button(
            roi_frame, text="Redraw", command=self.edit_roi, state='disabled')
        self.roi_btn_del = tk.Button(
            roi_frame, text="Delete", command=self.del_roi, state='disabled')

        self.widgets['regions'] = rois

        roi_frame.grid(row=7, sticky='w')
        roi_btn_add.pack(side='left', padx=15)
        rois.pack(side='left', padx=3)
        self.roi_btn_show.pack(side='left', padx=5)
        self.roi_btn_edit.pack(side='left', padx=5)
        self.roi_btn_del.pack(side='left', padx=5)

        # ------ OUTPUT SECTION ------

        output_frame = tk.LabelFrame(self, text="Output")
        outputdirframe = tk.Frame(output_frame)
        outputname = tk.Entry(
            outputdirframe, width=60, state='readonly')
        output_btn = tk.Button(outputdirframe, text="Browse",
                               command=self.find_outputfolder)
        outputinfo_frame = tk.Frame(output_frame)
        outputframes_label = tk.LabelFrame(outputinfo_frame,
                                           text="Output frames")
        outputframes = tk.Entry(outputframes_label, width=15)
        font_size = tk.LabelFrame(outputinfo_frame, text="Font size")
        font_size_entry = tk.Entry(font_size, width=15)

        self.widgets['save_as'] = outputname
        self.widgets['output_overlayed_images'] = outputframes
        self.widgets['font_size'] = font_size_entry

        output_frame.grid(row=8, sticky='w')
        outputdirframe.grid(row=0, column=0)
        outputinfo_frame.grid(row=1, column=0, sticky='w')


        outputname.grid(row=0, padx=10)
        output_btn.grid(row=0, column=1, padx=5)

        outputframes_label.grid(row=0, sticky='w', padx=10)
        outputframes.grid(row=0, column=0, sticky='w', padx=10)
        font_size.grid(row=0, column=1, sticky='w', padx=10)

        font_size_entry.pack(side='left', padx=3)
        # ------ Buttons ------
        add_job_btn_frame = tk.Frame(self)
        if editing:
            add_job_btn_text = "Confirm edits"
        else:
            add_job_btn_text = "Add job"
        add_job_btn = tk.Button(
            add_job_btn_frame, text=add_job_btn_text, command=self.add_job)
        cancel_btn = tk.Button(
            add_job_btn_frame, text="Cancel", command=self.destroy)

        add_job_btn_frame.grid(row=9, sticky='W')
        add_job_btn.pack(side='left', padx=5)
        cancel_btn.pack(side='left', padx=5)

        self.fill_values(self.edit_job)

    def fill_values(self, job=None):
        """Fill values with default or job if not None."""
        for key in self.widgets:
            widget = self.widgets[key]
            dtype = DEFAULTS[key][1]
            if job and key in job:
                value = dtype(job[key])
            elif DEFAULTS[key][0] is not None:
                value = dtype(DEFAULTS[key][0])
            else:
                continue
            if key == 'regions':
                # Region of interests:
                widget.configure(state='normal')
                if job:
                    self.regions = job[key]
                widget['values'] = value
                if len(value) > 0:
                    widget.set(value[0])
                    self.roi_btn_show.configure(state='normal')
                    self.roi_btn_edit.configure(state='normal')
                    self.roi_btn_del.configure(state='normal')
                widget.configure(state='readonly')
            elif key == 'video_filename':
                pass
            elif isinstance(widget, ttk.Combobox):
                widget.current(1 if value else 0)
            elif isinstance(widget, tk.Entry):
                readonly = False
                if widget.cget('state') == 'readonly':
                    widget.configure(state='normal')
                    readonly = True
                widget.delete(0, 'end')
                widget.insert(0, value)
                if readonly:
                    widget.configure(state='readonly')
            elif isinstance(widget, tk.IntVar):
                widget.set(value)
        if job:
            self.update_video_info(job['video_filename'])

    def add_to_job(self, job, fieldname, inputfield, typeconv):
        if isinstance(inputfield, typeconv):
            typed = inputfield
        else:
            if isinstance(inputfield, ttk.Combobox):
                string = inputfield.current()
            else:
                string = inputfield.get()
            if isinstance(string, int):
                typed = typeconv(string)
            else:
                if len(string) == 0:
                    err = "Field '" + fieldname + "' empty!"
                    tkinter.messagebox.showerror('Error', err)
                    return False
                try:
                    typed = typeconv(string)
                except Exception:
                    err = "Error in field '" + fieldname + "'"
                    tkinter.messagebox.showerror('Error', err)
                    return False
        job[fieldname] = typed
        return True

    def add_job(self):
        """Get values and load in job."""
        try:
            video = cv2.VideoCapture(self.widgets['video_filename'].get(
                ).split(", ")[0])
            ret, frame = video.read()
        except Exception:
            ret = False
        if not ret:
            self.parent.log('Select a movie first.')
            self.destroy()
            return
        if frame is None:
            self.parent.log('Corrupted movie.')
            self.destroy()
            return
        job = {}
        for key in self.widgets:
            if key == 'regions':
                job[key] = self.regions
            else:
                widget = self.widgets[key]
                dtype = DEFAULTS[key][1]
                self.add_to_job(job, key, widget, dtype)

        if self.editing:
            name = self.widgets['video_filename'].get().split('/')[-1]
            self.parent.job_buttons[self.edit_index]['thisframe'].pack_forget()
            self.parent.log('Edited job "%s".' % name)
            del self.parent.job_buttons[self.edit_index]
            del self.parent.jobs[self.edit_index]
        self.parent.add_job(job)
        self.destroy()

    def update_video_info(self, filename, filenames=None):
        if filenames is None:
            filenames = (filename,)
        try:
            video = cv2.VideoCapture(filename)
            n_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        except Exception:
            self.parent.log('Error opening video: ' + filename)
            return
        if n_frames < 0.5:
            self.parent.log('Error opening video: ' + filename)
            return
        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = video.get(cv2.CAP_PROP_FPS)

        self.widgets['video_filename'].config(state='normal')
        self.widgets['video_filename'].delete(0, tk.END)
        self.widgets['video_filename'].insert(0, ", ".join(filenames))
        self.widgets['video_filename'].config(state='readonly')
        self.videoinfo.config(text='Size: %dx%d    Number of frames: %d'
                              '    Frames per second guesstimate: %d'
                              % (width, height, n_frames, fps))
        self.widgets['limit_images_to'].delete(0, tk.END)
        self.widgets['limit_images_to'].insert(0, '%d' % n_frames)
        self.widgets['fps'].delete(0, tk.END)
        self.widgets['fps'].insert(0, '%d' % fps)

    def find_video(self):
        filenames = tkinter.filedialog.askopenfilenames()
        if len(filenames) == 0:
            return
        if "," in "".join(filenames):
            self.parent.log("Video paths cannot contain commas.")
            return
        filename = filenames[0]
        if filename:
            self.update_video_info(filename, filenames)
        self.grab_set()

    def add_roi(self):
        self.adding_roi = True
        roi = Roi(self)

    def show_roi(self):
        name = self.widgets['regions'].get()
        region = self.regions[name]
        try:
            video = cv2.VideoCapture(self.widgets['video_filename'].get().split(", ")[0])
            ret, frame = video.read()
        except Exception:
            ret = False
        if not ret:
            self.log('Select a movie first.')
            return
        try:
            start = int(self.widgets['start_frame'].get())
            end = int(self.widgets['limit_images_to'].get())
        except BaseException:
            start = 0
            end = video.get(cv2.CAP_PROP_FRAME_COUNT) - 1
        mid = int((end - start) // 2)
        video.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = video.read()

        plt.figure(figsize=(12, 9.5))
        plt.imshow(frame)
        plt.plot(region['x'], region['y'], 'b')
        plt.plot([region['x'][0], region['x'][-1]],
                 [region['y'][0], region['y'][-1]], 'b')
        axes = (0, frame.shape[1], frame.shape[0], 0)
        plt.axis(axes)
        plt.show()

    def del_roi(self):
        if tkinter.messagebox.askquestion(
                "Delete", "Are You Sure?", icon='warning'):
            name = self.widgets['regions'].get()
            self.widgets['regions']['values'] = [k for k in self.widgets['regions']['values'] if k != name]
            del self.regions[name]
            if len(self.widgets['regions']['values']) > 0:
                self.widgets['regions'].set(self.widgets['regions']['values'][0])
            else:
                self.widgets['regions'].set('')
                self.roi_btn_show.configure(state='disabled')
                self.roi_btn_edit.configure(state='disabled')
                self.roi_btn_del.configure(state='disabled')

    def edit_roi(self):
        self.adding_roi = False
        roi = Roi(self)

    def find_outputfolder(self):
        filename = tkinter.filedialog.asksaveasfilename(
            filetypes=[('', 'Directory name')])
        if filename:
            self.widgets['save_as'].config(state='normal')
            self.widgets['save_as'].delete(0, tk.END)
            self.widgets['save_as'].insert(0, filename)
            self.widgets['save_as'].config(state='readonly')


class Roi(tk.Toplevel):
    def closing(self):
        if len(self.xx) <= 2:
            self.destroy()
            self.parent.grab_set()
            return

        self.parent.widgets['regions'].configure(state='normal')
        self.parent.roi_btn_show.configure(state='normal')
        self.parent.roi_btn_edit.configure(state='normal')
        self.parent.roi_btn_del.configure(state='normal')

        if self.parent.adding_roi:
            name = None
            while not isinstance(name, str):
                name = tkinter.simpledialog.askstring(
                    'Region name', 'Input name of region', parent=self)
            self.parent.widgets['regions']['values'] = list(
                self.parent.widgets['regions']['values']) + [name]
            self.parent.widgets['regions'].set(name)
        else:
            name = self.parent.widgets['regions'].get()

        self.parent.regions[name] = {'x': self.xx, 'y': self.yy}

        self.parent.widgets['regions'].configure(state='readonly')
        self.destroy()
        self.parent.grab_set()

    def __init__(self, parent, *args, **kwargs):
        tk.Toplevel.__init__(self, *args, **kwargs)
        self.xx = []
        self.yy = []
        self.protocol('WM_DELETE_WINDOW', self.closing)
        self.wm_title("Region of Interest")
        self.grab_set()
        self.parent = parent

        self.f = matplotlib.figure.Figure(figsize=(12, 9.5))
        self.ax = self.f.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.f, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

        try:
            video = cv2.VideoCapture(parent.widgets['video_filename'].get(
                ).split(", ")[0])
            ret, frame = video.read()
        except Exception:
            ret = False
        if frame is None:
            parent.parent.log('Corrupted movie.')
            self.destroy()
            return
        if not ret:
            parent.parent.log('Select a movie first.')
            self.destroy()
            return
        try:
            start = int(parent.widgets['start_frame'].get())
            end = int(parent.widgets['limit_images_to'].get())
        except BaseException:
            start = 0
            end = video.get(cv2.CAP_PROP_FRAME_COUNT) - 1
        mid = int((end - start) // 2)
        video.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = video.read()
        if frame is None:
            parent.parent.log('Corrupted movie.')
            self.destroy()
            return
        self.ax.imshow(frame)
        axes = (0, frame.shape[1], frame.shape[0], 0)
        self.ax.axis(axes)
        poly = [1]

        def onclick(event):
            x, y = event.xdata, event.ydata
            if x is not None:
                self.xx.append(x)
                self.yy.append(y)
                self.ax.plot(self.xx, self.yy, '-xb')
                if len(self.xx) >= 3:
                    if poly[0] != 1:
                        poly[0].pop(0).remove()
                    poly[0] = self.ax.plot(
                        [self.xx[0], self.xx[-1]],
                        [self.yy[0], self.yy[-1]], '--b')
                self.ax.axis(axes)
                self.canvas.draw()
        cid = self.canvas.mpl_connect('button_press_event', onclick)


def run():
    """Run app"""
    faulthandler.enable()
    root = tk.Tk()
    MainApplication(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
