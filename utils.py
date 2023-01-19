from IPython import display
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def load_src_trg(source, target, sequence_length, offset, batch_size=1):
    dataset = SequenceDataset(source, target, sequence_length, offset)
    src_trg = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return src_trg


class SequenceDataset(Dataset):
    def __init__(self, source, target, sequence_length=5, offset=0):
        self.sequence_length = sequence_length
        self.offset = offset
        self.source = source
        self.target = target

    def __len__(self):
        return self.source.shape[1] - self.sequence_length - self.offset + 2

    def __getitem__(self, i):
        i = i if i + self.sequence_length + self.offset < (self.source.shape[1] - self.sequence_length - self.offset + 1) else 0
        src = self.source[:, i:i + self.sequence_length]
        trg = self.target[:, i + 1:i + self.sequence_length + self.offset]
        trg_y = self.target[:, i + 1:i + self.sequence_length + self.offset]
        return src, trg, trg_y


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    def __init__(self,
                 xlabel=None,
                 ylabel=None,
                 legend=None,
                 xlim=None,
                 ylim=None,
                 xscale='linear',
                 yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'),
                 nrows=1,
                 ncols=1,
                 figsize=(5, 3)):
        if legend is None:
            legend = []

        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)

        if nrows * ncols == 1:
            self.axes = [self.axes, ]

        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
            self.axes[0].cla()
            for x, y, fmt in zip(self.X, self.Y, self.fmts):
                self.axes[0].plot(x, y, fmt)
            self.config_axes()
            display.display(self.fig)
            display.clear_output(wait=True)


def set_axes(axes,
             xlabel,
             ylabel,
             xlim,
             ylim,
             xscale,
             yscale,
             legend):
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

