import pickle
from tkinter import Button, Tk, Canvas, ALL, messagebox
from PIL import ImageGrab
import matplotlib.pyplot as plt
import numpy as np


SIZE = 30 # this size is used as im pretty sure it is the best size for the model to classify the digit

class Model:
    model = pickle.load(open("neural-network-model/model.pickle", "rb"))

    @classmethod 
    def predict(cls, image): # called to predict the number which has been drawn on the canvas
        predictions = cls.model.predict(np.array([image]))
        return np.argmax(predictions)
        
class DrawingApp:
    def __init__(self):
        self.root = Tk()
        self.root.title("Handwritten Digit Predictor")
        self.root.geometry("500x500")
        self.root.resizable(False,False)
        self.canvas = Canvas(self.root, width=450, height=450 ,background="#FFFFFF")
        self.reset = Button(self.root, text="Reset", command=self.reset)
        self.predict = Button(self.root, text="Predict", command=self.predict_process)
        self.reset.pack(side="bottom")
        self.predict.pack(side="bottom")
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.pack()
        self.root.mainloop()

    def draw(self, event): # function to allow drawing when the mouse is clicked
        x, y = event.x, event.y
        self.canvas.create_oval(x, y, x + SIZE, y + SIZE, fill="black")

    def reset(self):
        self.canvas.delete(ALL)

    def predict_process(self): # preprocessing the image so it can be classified by the model
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        image = ImageGrab.grab().crop((x,y,x1,y1))
        image = np.abs((np.array(image.resize((28,28)).convert("L")) / 255) - 1)
        prediction = Model.predict(image)
        messagebox.showinfo(title="Prediction", message=f"Model predicted: {prediction}")

def main():
    app = DrawingApp()

if __name__ == "__main__":
    main()