from os import listdir
from os.path import isfile, join
import sys


class Position(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Folder:
    def __init__(self, path):
        self.path = path

    def files(self, absolute=True):
        files = [f for f in listdir(self.path) if isfile(join(self.path, f)) and not f.endswith('.py')]
        if len(files) == 0:
            raise IOError
        if absolute:
            files = [self.path + i for i in files]
            return files
        if not absolute:
            return files


class Filename(str):
    def filename(self):
        temp = self.split('/')[-1].lower()
        temp = ''.join(temp.split('.')[:-1])
        return temp

    def v(self):
        vel = None
        for i in self.filename().split('_'):
            if 'ms' in i:
                i = i.replace('ms', '')
                vel = int(i)
                return vel

    def angle(self):
        ang = None
        for i in self.filename().split('_'):
            try:
                i = i.replace('sinus', '')
            except:
                pass
            if 'sin' in i:
                i = i.replace('sin', '')
                if not i == 'us':
                    ang = int(i)
                    return ang
            if 'deg' in i:
                i = i.replace('deg', '')
                ang = int(i)
                return ang
            if 'grad' in i:
                i = i.replace('grad', '')
                ang = int(i)
                return ang

    def frequency(self):
        f = None
        for i in self.filename().split('_'):
            if 'hz' in i:
                i = i.replace('hz', '')
                f = int(i)
                return f

    def position(self):
        posx = None
        posy = None
        posz = None
        for i in self.filename().split('_'):
            if 'posx' in i:
                i = i.replace('posx', '')
                posx = int(i)
            if 'posy' in i:
                i = i.replace('posy', '')
                posy = int(i)
            if 'posz' in i:
                i = i.replace('posz', '')
                posz = int(i)

            if 'xpos' in i:
                i = i.replace('xpos', '')
                posx = int(i)
            if 'ypos' in i:
                i = i.replace('ypos', '')
                posy = int(i)
            if 'zpos' in i:
                i = i.replace('zpos', '')
                posz = int(i)
        return Position(posx, posy, posz)

    def number(self):
        pos = None

        for i in self.filename().split('_'):
            if 'posx' in i:
                i = i.replace('posx', '')
            if 'posy' in i:
                i = i.replace('posy', '')
            if 'posz' in i:
                i = i.replace('posz', '')
            if 'xpos' in i:
                i = i.replace('xpos', '')
            if 'ypos' in i:
                i = i.replace('ypos', '')
            if 'zpos' in i:
                i = i.replace('zpos', '')
            if 'pos' in i:
                i = i.replace('pos', '')
                pos = int(i)
        return pos


def Plot(ax, data, errorbars=False, **kwargs):
    title = kwargs.pop('title', '')
    xlim = kwargs.pop('xlim', None)
    ylim = kwargs.pop('ylim', None)
    xlabel = kwargs.pop('xlabel', '')
    ylabel = kwargs.pop('ylabel', '')
    xscale = kwargs.pop('xscale', 'linear')
    yscale = kwargs.pop('yscale', 'linear')
    if not isinstance(data, list):
        label = kwargs.pop('label', data.label)
    labels = kwargs.pop('labels', None)

    print 'plotting'

    if isinstance(data, list):
        for ind, i in enumerate(data):
            if i.yerror is None or errorbars is False:
                plot_func = ax.plot
            if not i.yerror is None and errorbars is True:
                plot_func = ax.errorbar
                kwargs['yerr'] = i.yerror
                kwargs['ecolor'] = 'yellow'

            if labels is None:
                out = plot_func(i.x, i, label=i.label, **kwargs)
            else:
                out = plot_func(i.x, i, label=labels[ind], **kwargs)
    else:
        if data.yerror is None or errorbars is False:
            plot_func = ax.plot

        if not data.yerror is None and errorbars is True:
            plot_func = ax.errorbar
            kwargs['yerr'] = data.yerror
            kwargs['ecolor'] = 'yellow'
            kwargs['alpha'] = 0.5
        out = plot_func(data.x, data, label=label, **kwargs)

    ax.autoscale()
    ax.grid()
    ax.set_title(title)
    # if self.xlim is None and self.ylim is None:
    #   self.ax.autoscale()

    if isinstance(xlim, tuple):
        ax.set_xlim(xlim)
    if isinstance(ylim, tuple):
        ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.grid()
    out = ax.legend(loc='best', fancybox=True, framealpha=0.5)
    return out


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
