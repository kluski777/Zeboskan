import matplotlib.pyplot as plt
import numpy as np

def display_images_size_compared(*batches, titles=None):
    n = len(batches)
    titles = titles or [f"Batch {i+1}" for i in range(n)]

    for items in zip(*batches):
        ref_img = items[0][0]
        if ref_img.shape[1] >= 2100: continue

        fig, axes = plt.subplots(1, n, figsize=(10 * n, 10))
        axes = [axes] if n == 1 else axes

        for ax, item, t in zip(axes, items, titles):
            img, meta = item[0], item[1] # Manual index fixes unpack error

            ax.imshow(img)
            ax.set_title(f"{t} {img.shape[:2]}")
            ax.axis('off')
            
            # Setup Scaling (W, H)
            dims = np.array(img.shape[:2][::-1])
            ratio = dims / np.array(ref_img.shape[:2][::-1])

            for obj in meta.get('objects', []):
                p = obj['points']['exterior']
                p = np.array(p + [p[0]]) # Close loop
                
                # Rescale if points don't fit in the current image size
                if (p > dims).any() and (img is not ref_img):
                    p = p * ratio
                    
                ax.plot(p[:, 0], p[:, 1], c='lime', lw=2)
                
        plt.tight_layout()
        plt.show()

def show_images(images, cols=4, title=None):
    n = len(images)
    rows = (n + cols - 1) // cols
    
    if cols == 1:
        fig, axes = plt.subplots(rows, 1, figsize=(5, rows*5))
    else:
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    
    if title:
        fig.suptitle(title, fontsize=16)
    axes = np.array(axes).flatten() if isinstance(axes, np.ndarray) else [axes]
    for ax, img in zip(axes, images):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    for ax in axes[n:]:
        ax.remove()
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.show()

# jeszcze jakiegos jebitnego refactora jebnac wypadaloby