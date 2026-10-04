import matplotlib as mpl
import matplotlib.pyplot as plt


def update_rcparams(high_dpi: bool = True, scale: float = 2.0, dpi: int = 130):
    """Configures matplotlib rcParams for crisp, readable display on modern high-DPI monitors.

    Args:
        high_dpi (bool): Whether to use enlarged high-DPI settings (default True).
        scale (float): GUI scaling factor for TkAgg toolbar buttons and window chrome.
        dpi (int): Canvas dots-per-inch for rendering.
    """
    if high_dpi:
        mpl.rcParams['figure.dpi'] = dpi
        mpl.rcParams['font.size'] = 11
        mpl.rcParams['axes.titlesize'] = 13
        mpl.rcParams['axes.labelsize'] = 12
        mpl.rcParams['xtick.labelsize'] = 10
        mpl.rcParams['ytick.labelsize'] = 10
        mpl.rcParams['legend.fontsize'] = 10
        mpl.rcParams['figure.titlesize'] = 15
        mpl.rcParams['lines.linewidth'] = 1.8
    else:
        mpl.rcParams['font.size'] = 8
        mpl.rcParams['axes.titlesize'] = 8
        mpl.rcParams['xtick.labelsize'] = 6
        mpl.rcParams['ytick.labelsize'] = 6

    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False


def configure_high_dpi_window(fig, scale: float = 2.0):
    """Enlarges TkAgg navigation toolbar buttons and window scaling on high-res monitors.

    Call this right before plt.show().
    """
    try:
        manager = fig.canvas.manager
        if hasattr(manager, 'window') and hasattr(manager.window, 'tk'):
            manager.window.tk.call('tk', 'scaling', scale)
        if hasattr(manager, 'toolbar') and manager.toolbar is not None:
            import tkinter as tk
            for child in manager.toolbar.winfo_children():
                try:
                    img_name = child.cget('image')
                    if img_name:
                        photo = tk.PhotoImage(name=img_name)
                        zoomed = photo.zoom(int(scale), int(scale))
                        child.config(image=zoomed)
                        child._zoomed_image = zoomed
                except Exception:
                    pass
                try:
                    child.config(font=('TkDefaultFont', 11))
                except Exception:
                    pass
    except Exception:
        pass
