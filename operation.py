import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
class Custering :
    def kmeans_calculate(self, x, k, mode='manhattan'):
        x = x.astype(np.float64)
        centroids = x[np.random.permutation(len(x))[:k]]
        while True:
            D = np.zeros((len(x), k))
            for i in range(k):
                if mode == 'manhattan':
                    D[:, i] = np.mean(np.abs(x - centroids[i]), axis=1)
                elif mode == 'square_euclidean':
                    D[:, i] = np.mean((x - centroids[i])**2, axis=1)

            idx = np.argmin(D, axis=1)
            c_old = centroids.copy()
            for i in range(k):
                if np.any(idx == i):
                    centroids[i] = np.mean(x[idx == i], axis=0)

            change = np.mean(np.abs(c_old - centroids))
            print(f"Change: {change}")
            if change < 1e-6:
                break
    
        return idx

    def group_custering(self, file_names, labels, img_source_folder, cluster_output_path):
        os.makedirs(cluster_output_path, exist_ok=True)
        for name, label in zip(file_names, labels):
            cluster_folder = os.path.join(cluster_output_path, f"cluster_{label}")
            os.makedirs(cluster_folder, exist_ok=True)
            img = cv2.imread(os.path.join(img_source_folder, name))
            if img is None:
                continue
            cv2.imwrite(os.path.join(cluster_folder, name), img)

    def visualize(self, folder_output, thumb_h=170, cell_w=190):
        """One scrollable matplotlib window per cluster, all shown together.

        Thumbnails are pasted into one mosaic per window; scrolling (wheel, keys or
        slider) only moves the y-limits, and resizing only re-flows the columns."""
        image_support = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')

        groups = []  # (cluster_name, [(label, thumb_rgb), ...])
        for sub_folder in sorted(os.listdir(folder_output)):
            subfolder_path = os.path.join(folder_output, sub_folder)
            if not os.path.isdir(subfolder_path):
                continue
            members = []
            for f in sorted(os.listdir(subfolder_path)):
                if not f.lower().endswith(image_support):
                    continue
                img = cv2.imread(os.path.join(subfolder_path, f))
                if img is None:
                    print(f"Error loading {os.path.join(subfolder_path, f)}")
                    continue
                h, w = img.shape[:2]
                scale = min(thumb_h / h, (cell_w - 10) / w)  # fit inside the cell
                img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                members.append((os.path.splitext(f)[0], img))
            if members:
                groups.append((sub_folder, members))
        if not groups:
            print("No images found in the specified folder.")
            return

        pad, header_h = 12, 44
        cell_h = thumb_h + 28  # thumbnail + label
        keep_alive = []        # Slider widgets stop responding if garbage-collected

        def make_cluster_window(index, name, members):
            title = f"{name.replace('_', ' ').title()}  ({len(members)} images)"
            fig = plt.figure(figsize=(cell_w * 5 / 100 + 0.4, 6), dpi=100)
            fig.canvas.manager.set_window_title(title)
            try:  # cascade the windows so they do not hide each other
                win = fig.canvas.manager.window
                if hasattr(win, 'wm_geometry'):      # TkAgg
                    win.wm_geometry(f"+{40 + index * 45}+{40 + index * 45}")
                elif hasattr(win, 'move'):           # Qt
                    win.move(40 + index * 45, 40 + index * 45)
            except Exception:
                pass

            ax = fig.add_axes([0, 0, 0.975, 1])
            slider_ax = fig.add_axes([0.982, 0.03, 0.012, 0.94])
            slider = Slider(slider_ax, '', 0.0, 1.0, valinit=1.0, orientation='vertical')
            slider.valtext.set_visible(False)
            keep_alive.append(slider)

            state = {'cols': 0, 'height': 1, 'width': 1, 'top': 0.0}

            def view_height():
                bbox = ax.get_window_extent()
                return state['width'] * bbox.height / max(bbox.width, 1)

            def update_view():
                vh = view_height()
                max_top = max(0.0, state['height'] - vh)
                state['top'] = min(max(state['top'], 0.0), max_top)
                ax.set_ylim(state['top'] + vh, state['top'])  # y grows downward
                slider.set_val(1.0 - state['top'] / max_top if max_top > 0 else 1.0)
                fig.canvas.draw_idle()

            def build(cols):
                """Paste all thumbnails into one mosaic for the given column count."""
                rows = int(np.ceil(len(members) / cols))
                total_h, total_w = header_h + rows * cell_h + pad, cols * cell_w + 2 * pad
                mosaic = np.full((total_h, total_w, 3), 255, np.uint8)  # plain white
                ax.clear()
                ax.axis('off')
                ax.text(pad, header_h / 2, title, fontsize=15,
                        fontweight='bold', va='center', ha='left', clip_on=True)
                for i, (label, thumb) in enumerate(members):
                    r, c = divmod(i, cols)
                    th, tw = thumb.shape[:2]
                    x = pad + c * cell_w + (cell_w - tw) // 2
                    ty = header_h + r * cell_h
                    mosaic[ty:ty + th, x:x + tw] = thumb
                    ax.text(pad + c * cell_w + cell_w / 2, ty + thumb_h + 14, label, fontsize=8,
                            ha='center', va='center', clip_on=True)
                ax.imshow(mosaic, aspect='auto', interpolation='antialiased')
                state.update(cols=cols, height=total_h, width=total_w)

            def fit_columns():
                cols = max(1, int(ax.get_window_extent().width // cell_w))
                if cols != state['cols']:
                    frac = state['top'] / state['height']
                    build(cols)
                    state['top'] = frac * state['height']  # keep roughly the same scroll position

            def on_resize(*_):
                fit_columns()
                update_view()

            def on_scroll(event):
                state['top'] -= event.step * view_height() * 0.12
                update_view()

            def on_key(event):
                vh = view_height()
                moves = {'up': -vh * 0.1, 'down': vh * 0.1, 'pageup': -vh * 0.9, 'pagedown': vh * 0.9,
                         'home': -state['height'], 'end': state['height']}
                if event.key in moves:
                    state['top'] += moves[event.key]
                    update_view()

            def on_slider(val):
                vh = view_height()
                state['top'] = (1.0 - val) * max(0.0, state['height'] - vh)
                ax.set_ylim(state['top'] + vh, state['top'])
                fig.canvas.draw_idle()

            fig.canvas.mpl_connect('resize_event', on_resize)
            fig.canvas.mpl_connect('scroll_event', on_scroll)
            fig.canvas.mpl_connect('key_press_event', on_key)
            slider.on_changed(on_slider)

            fit_columns()
            update_view()

        for index, (name, members) in enumerate(groups):
            make_cluster_window(index, name, members)

        plt.show()
