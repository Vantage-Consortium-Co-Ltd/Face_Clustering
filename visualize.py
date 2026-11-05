import os
import matplotlib.pyplot as plt


class Visualization:

    def visualize(self, folder_output, cols=5):
        """One matplotlib window per cluster folder, all shown together."""
        image_support = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
        for index, sub_folder in enumerate(sorted(os.listdir(folder_output))):
            folder = os.path.join(folder_output, sub_folder)
            if not os.path.isdir(folder):
                continue
            files = sorted(f for f in os.listdir(folder) if f.lower().endswith(image_support))
            if not files:
                continue

            rows = -(-len(files) // cols)  # ceil division
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.6 + 0.6), squeeze=False)
            fig.suptitle(f"{sub_folder.replace('_', ' ').title()}  ({len(files)} images)", fontweight='bold')
            for ax in axes.ravel():
                ax.axis('off')
            for ax, f in zip(axes.ravel(), files):
                ax.imshow(plt.imread(os.path.join(folder, f)))
                ax.set_title(os.path.splitext(f)[0], fontsize=8)

            offset = 40 + index * 45  # cascade windows so they do not hide each other
            try:  # shortcut: Tk and Qt backends only, others keep the default position
                win = fig.canvas.manager.window
                if hasattr(win, 'wm_geometry'):
                    win.wm_geometry(f"+{offset}+{offset}")
                else:
                    win.move(offset, offset)
            except Exception:
                pass

        plt.show()
