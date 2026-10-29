"""Writing GIF animations with Matplotlib and Pillow."""

import os

from matplotlib.animation import FuncAnimation, PillowWriter

from .io_utils import ANIMATIONS, ensure_dirs
from .plotting import plt


def save_gif(fig, update, frames, name, fps=15, dpi=110):
    """Render ``update(i)`` for each frame into outputs/animations."""
    ensure_dirs()
    animation = FuncAnimation(fig, update, frames=frames, blit=False)
    path = os.path.join(ANIMATIONS, f"{name}.gif")
    animation.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path
