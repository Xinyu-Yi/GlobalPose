r"""
    Colorbar which evaluate the color for a specified value.
"""


__all__ = ['ColorBar']


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


class ColorBar:
    r"""
    Colorbar which evaluate the color for a specified value.
    """
    colormaps = list(mpl.colormaps)

    def __init__(self, vrange=(0, 1), cmap='jet'):
        r"""
        Init a colorbar.

        :param vrange: Value range for the colorbar.
        :param cmap: Matplotlib colormap. See ColorBar.colormaps for available colormaps.
        """
        self.vmin, self.vmax = vrange
        self.cmap = plt.get_cmap(cmap)

    def show(self, cmap_list=None):
        r"""
        Show the colorbar.

        :param cmap_list: List of colorbar names to show. Default is None, which shows the current colormap.
                          Use ColorBar.colormaps to get available colormaps.
        """
        if cmap_list is None:
            cmap_list = [self.cmap.name]

        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))
        nrows = len(cmap_list)
        figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
        fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
        fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                            left=0.2, right=0.99)
        for ax, name in zip(axs, cmap_list):
            ax.imshow(gradient, aspect='auto', cmap=mpl.colormaps[name])
            ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
                    transform=ax.transAxes)
        for ax in axs:
            ax.set_axis_off()
        plt.show()

    def evaluate(self, value):
        r"""
        Evaluate the color for specified values.

        :param value: Float or list/numpy/torch array.
        :return: Numpy array in shape [3] or [n, 3] for float RGB color in (0, 1).
        """
        value = np.array(value)
        t = (value - self.vmin) / (self.vmax - self.vmin)
        color = np.array(self.cmap(t))[..., :3]
        return color

    def __call__(self, value):
        return self.evaluate(value)


if __name__ == '__main__':
    bar = ColorBar()
    bar.show()
    bar.show(ColorBar.colormaps[:40])
    bar.show(ColorBar.colormaps[40:80])
    bar.show(ColorBar.colormaps[80:120])
    bar.show(ColorBar.colormaps[120:])
