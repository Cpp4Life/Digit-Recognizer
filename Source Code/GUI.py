from tkinter import*
from PIL import ImageGrab, Image
from tkinter import filedialog
from keras.models import load_model
import tkinter as tk
import win32gui
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from myMetrics import *

model = load_model('model2_mnist.h5', custom_objects={
                   'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})


def predict_digit(img):
    # resize image to 28x28 pixels
    img = img.resize((28, 28))
    # convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img, dtype='float32')
    height, width = (28, 28)
    for i in range(height):
        for j in range(width):
            img[i, j] = 255 - img[i, j]
    # reshaping to fit the model
    img = img.reshape(1, 28, 28, 1)
    img = img / 255
    # make predictions
    result = model.predict(img)[0]
    return np.argmax(result), max(result)


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Creating elements
        self.title('Handwritten Digit Recognizer')
        self.canvas = tk.Canvas(
            self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Waiting...", font=("Caladea", 48))
        self.classify_btn = tk.Button(
            self, text="Recognise", width=10, command=self.classify_handwriting, font=("Caladea", 13, 'bold'))
        self.button_clear = tk.Button(
            self, text="Clear", width=10, command=self.clear_all, font=("Caladea", 13, 'bold'))
        self.img_btn = tk.Button(
            self, text="Select an image", command=self.classify_image, font=("Caladea", 13, 'bold'))

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        self.img_btn.grid(row=2, pady=10)

        # self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        Handwriting = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(Handwriting)
        image = ImageGrab.grab(rect)
        digit, acc = predict_digit(image)
        self.label.configure(text='Digit: ' + str(digit) + '\n' + 'Percentage: ' +
                             str(int(acc * 100)) + '%')

    def classify_image(self):
       # Get user input
        _myImage = filedialog.askopenfilename(
            initialdir="./Images", filetypes=(("png files", "*.png"), ("all files", "*.*")))
        img = cv.imread(_myImage)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Show Image
        cv.namedWindow('Digit', cv.WINDOW_NORMAL)
        cv.resizeWindow('Digit', 500, 500)
        cv.imshow('Digit', img)
        # Fix and recognize Image
        gray = cv.resize(gray, (28, 28))
        gray = gray.astype('float32')
        gray = gray.reshape(1, 28, 28, 1)
        gray /= 255
        result = model.predict(gray)[0]
        digit, acc = np.argmax(result), max(result)
        self.label.configure(text='Digit: ' + str(digit) + '\n' + 'Percentage: ' +
                             str(int(acc * 100)) + '%')
      # image = ImageGrab.grab(cv.imshow('image', img))

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r,
                                self.x + r, self.y + r, fill="black")


# main func
if __name__ == '__main__':
    app = App()
    mainloop()
