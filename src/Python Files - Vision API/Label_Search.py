#Vision Example 2 - Label Search

#This code allows the user to enter a search term
# and compares it against the labels identified
#by the Vision API. If there is a match, it is
#displayed in the GUI.

import io
import os

# Imports the GUI library
import tkinter as tkinter

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

# This subroutine defines what will happen when the button is clicked
def btnclick():
    if ent.get() == "":
        btn.configure(text="No Filename")
    else:
        lbl6.configure(text="Uploading")
        # Initiates a client
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

        if searchent.get() != 0:
            lbl6.configure(text="Not found")

        #Match each identified label with the one being searched for
        for label in labels:
            #If there is a match, lbl6 is updated
            if label_search(label.description, searchent.get()) == True:
                lbl6.configure(text = "Label found")

        btn.configure(text="Done")
        btn.configure(text="Upload Image")

#Compares label with label to be searched for
#Returns True if there is a match
def label_search(label, search):
    if label == search:
        return True

# Instantiate a new GUI Window
window = tkinter.Tk()
window.title("Google Cloud Vision API")
window.geometry("350x250")
window.configure(background = "#ffffff")

#Defines GUI Elements
lbl = tkinter.Label(window, text="Google Cloud Vision API", fg="#383a39", bg="#ffffff", font=("Helvetica", 23))
lbl3 = tkinter.Label(window, text="")
lbl2 = tkinter.Label(window, text="Enter an image's filename and click 'Upload Image'")
ent = tkinter.Entry(window)
btn = tkinter.Button(window, text="Upload Image", command = btnclick)
# GUI for search term
lbl4 = tkinter.Label(window, text="")
lbl5 = tkinter.Label(window, text = "Search Term")
searchent = tkinter.Entry(window)
lbl6 = tkinter.Label(window, text = "")

#Packs GUI Elements into window
lbl.pack()
lbl3.pack()
lbl2.pack()
ent.pack()
btn.pack()
#Search graphics
lbl4.pack()
lbl5.pack()
searchent.pack()
lbl6.pack()

window.mainloop()
