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
import pickle
import tkinter.filedialog


def plot_path(filename):
    """Get path for saved trace and plot it."""
    saved_name = "/".join(filename.split('/')[:-1])

    plt.close()
    plt.figure(figsize=(10, 8))
    colormap = cm.Set2

    with open(filename, 'br') as f:
        # https://docs.python.org/3/library/pickle.html#pickle.loads
        # Using encoding='latin1' is required for unpickling
        # NumPy arrays and instances of datetime, date and time
        # pickled by Python 2.
        track = pickle.load(f, encoding='latin-1')

    particles = set(track['particle'])
    colours = [colormap(i / float(len(particles)))
               for i in range(len(particles))]
    rand = np.random.permutation(len(particles))
    for i, p in enumerate(particles):
        idx = track['particle'] == p

        x = track['x'][idx]
        y = track['y'][idx]

        plt.plot(y, x, c=colours[rand[i]], linewidth=2.5)
    plt.axis('equal')
    a = list(plt.axis())
    a[3], a[2] = a[2], a[3]
    plt.axis(a)

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.savefig(saved_name + '/track.png')
    plt.savefig(saved_name + '/track.pdf')

    plt.show()


if __name__ == '__main__':
    filename = tkinter.filedialog.askopenfilename(
        title='Locate a track.p file', filetypes=[("track.p file", "*.p")])
    plot_path(filename)
