import matplotlib.pyplot as plt
import cv2

def show_before_after(img1, img2, title1="Original", title2="Enhanced"):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title(title1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(title2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    plt.axis("off")

    plt.show()
