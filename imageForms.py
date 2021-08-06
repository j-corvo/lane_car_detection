
import cv2 as cv

from matplotlib import pyplot as plt
from   tkinter import *
from  tkinter import messagebox


#Show two images side by side
# receives the images and a BGR boolean to specify if the image is BGR or RGB format
def showSideBySideImages(img1, img2, title="", BGR1=True, BGR2=True):
    img1 = img1 if BGR1 else cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    img2 = img2 if BGR2 else cv.cvtColor(img2, cv.COLOR_BGR2RGB)
    fig = plt.figure(title)
    ax = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(img1)
    ax.set_title('Img1')
    plt.axis("off")
    if (len(img1.shape) < 3): #grayscale
        imgplot.set_cmap('gray')

    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(img2)
    ax.set_title('Img2')
    plt.axis("off")
    if (len(img2.shape) <3): #grayscale
        imgplot.set_cmap('gray')
    plt.show()


def showImage(img1):
    fig = plt.figure('Img1')
    imgplot = plt.imshow(img1)

    plt.axis("off")


def showMessageBox(title, message):
    app = Tk()
    app.withdraw()
    messagebox.showinfo(title, message=message)
