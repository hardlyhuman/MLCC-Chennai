#Vision API Example 1 - Label Return

#This code will display a GUI interface asking for
#the filename of an image. It will be sent and
#processed by the Vision API. A list of potential
#labels will be returned
import io
import os

# Imports the GUI library
import Tkinter as tkinter

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

#This subroutine defines what happens when the button is pressed
def btnclick():
    if ent.get() == "":
        btn.configure(text="No Filename")
    else:
        btn.configure(text="Uploading")
        # Instantiates a client
        client = vision.ImageAnnotatorClient()

        filename = ent.get()

        # The name of the image file to annotate
        file_name = os.path.join(
            os.path.dirname(__file__), filename)

        # Loads the image into memory
        with io.open(file_name, 'rb') as image_file:
            content = image_file.read()

        image = types.Image(content=content)

        # Performs label detection on the image file
        response = client.label_detection(image=image)
        labels = response.label_annotations

        #Match each identified label with the one being searched for
        for label in labels:
            l = tkinter.Label(window, text=label.description)
            l.pack()

        btn.configure(text="Done")
        btn.configure(text="Upload Image")

# Instantiate a new GUI Window
window = tkinter.Tk()
window.title("Google Cloud Vision API")
window.geometry("350x300")
window.configure(background = "#ffffff")

#Defines GUI Elements
lbl = tkinter.Label(window, text="Google Cloud Vision API", fg="#383a39", bg="#ffffff", font=("Helvetica", 23))
lbl3 = tkinter.Label(window, text="")
lbl2 = tkinter.Label(window, text="Enter an image's filename and click 'Upload Image'")
ent = tkinter.Entry(window)
btn = tkinter.Button(window, text="Upload Image", command = btnclick)

#Packs GUI Elements into window
lbl.pack()
lbl3.pack()
lbl2.pack()
ent.pack()
btn.pack()

window.mainloop()
