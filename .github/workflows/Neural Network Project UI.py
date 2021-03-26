import tkinter as tk
from tkinter import ttk
import tkinter.font as font
from tkinter import IntVar, StringVar

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from matplotlib import style

from PIL import ImageTk,Image

import string

import network

# ------------------------ PAGES ------------------------

# Initialise Home Page

root = tk.Tk()
root.title("NetTest - Home Page")
root.iconbitmap("FeatherflyCircleLogoIcon.ico")
root.geometry("800x600")
root.resizable(False, False)

# Display NetTest Banner
img_netTest = ImageTk.PhotoImage(Image.open("images/netTest.png").resize((650,320), Image.ANTIALIAS))
lbl_img_netTest = tk.Label(image=img_netTest)
lbl_img_netTest.grid(row=1,column=0,columnspan=3,pady=(30,0))



# ------------------------ CREATE MODEL PAGE ------------------------

def pageCreateModel():

    # Initialise Create Model Page
    winCreateModel = tk.Toplevel()
    winCreateModel.title("NetTest - Create a model")
    winCreateModel.iconbitmap("FeatherflyCircleLogoIcon.ico")
    winCreateModel.geometry("800x600")
    winCreateModel.resizable(False, False)

    # Create Options Frame
    frame_options_create = tk.Frame(winCreateModel, width=300,height=500) 
    frame_options_create.grid(row=1,column=0,padx=10,pady=10)

    frame_results = tk.Frame(winCreateModel, width=300,height=500)
    frame_results.grid(row=1,column=1,padx=10,pady=10)

    # Variables
    isValidation = IntVar()


    # Create Labels
    lbl_numHidden = tk.Label(frame_options_create,text="Number of Neurons in Hidden Layer",font=fontLabel)
    lbl_numHidden.grid(row=1,column=0,sticky="ew")

    lbl_numEpochs = tk.Label(frame_options_create,text="Number of Epochs to Train for",font=fontLabel)
    lbl_numEpochs.grid(row=3,column=0,sticky="ew")

    lbl_miniBatch = tk.Label(frame_options_create,text="Mini-batch Size - 1 is Online Learning",font=fontLabel)
    lbl_miniBatch.grid(row=5,column=0,sticky="ew")

    lbl_eta = tk.Label(frame_options_create,text="Learning Rate",font=fontLabel)
    lbl_eta.grid(row=7,column=0,sticky="ew")
    

    # Create Sliders
    sld_numHidden = tk.Scale(frame_options_create,from_=0,to=1000,orient="horizontal")
    sld_numHidden.grid(row=2,column=0,sticky="ew")

    sld_numEpochs = tk.Scale(frame_options_create,from_=1,to=1000,orient="horizontal")
    sld_numEpochs.grid(row=4,column=0,sticky="ew")

    sld_miniBatch = tk.Scale(frame_options_create,from_=1,to=1000,orient="horizontal")
    sld_miniBatch.grid(row=6,column=0,sticky="ew")

    sld_eta = tk.Scale(frame_options_create,from_=0.0001,to=10,orient="horizontal",resolution=0.0001)
    sld_eta.grid(row=8,column=0,sticky="ew")

    cbtn_validation = tk.Checkbutton(frame_options_create,text="Use Validation Data",variable=isValidation, font=fontLabel)
    cbtn_validation.grid(row=9,column=0,pady=(20,0),sticky="ew")

    # Create Buttons
    btn_trainModel = tk.Button(frame_options_create, text="Train Model", command=lambda:trainModel(winCreateModel,frame_results,isValidation.get(),int(sld_numHidden.get()),int(sld_numEpochs.get()),int(sld_miniBatch.get()),float(sld_eta.get())), font=fontButton, width=30,height=3,bg="#D5DBDB")    
    btn_trainModel.grid(row=10,column=0,padx=(20,20),pady=(40,0),sticky="nsew")
    
    winCreateModel.update()

# ------------------------ CREATE SEARCH PAGE ------------------------

def pageSearchModel():
    # Initialise Create Model Page
    winSearchModel = tk.Toplevel()
    winSearchModel.title("NetTest - Model Search")
    winSearchModel.iconbitmap("FeatherflyCircleLogoIcon.ico")
    winSearchModel.geometry("1200x600")
    winSearchModel.columnconfigure(0, weight=1)

    # Create Frames
    frame_form_search = tk.Frame(winSearchModel, width=1200,height=200) 
    frame_form_search.grid(row=0,column=0,padx=10,pady=10,sticky="nsew")

    frame_results_search = tk.Frame(winSearchModel, width=1200,height=400)
    frame_results_search.grid(row=1,column=0,padx=10,sticky="nsew")
    frame_results_search.columnconfigure(0, weight=1)

    
    # Create Labels
    lbl_searchID = tk.Label(frame_form_search,text="Enter Model ID:",anchor="w",font=fontLabel)
    lbl_searchID.grid(row=1,column=0,columnspan=2,sticky="ew")
    
    lbl_searchName = tk.Label(frame_form_search,text="Enter Model Name:",anchor="w",font=fontLabel)
    lbl_searchName.grid(row=3,column=0,columnspan=2,sticky="ew")

    lbl_searchAccuracy = tk.Label(frame_form_search,text="Enter Model Accuracy:",anchor="w",font=fontLabel)
    lbl_searchAccuracy.grid(row=5,column=0,columnspan=2,sticky="ew")

    lbl_numResults = tk.Label(frame_form_search,text="",font=fontLabel)
    lbl_numResults.grid(row=8,column=0,columnspan=2,sticky="w")

    # Create Entries
    ent_searchID = tk.Entry(frame_form_search,width=30,font=fontLabel)
    ent_searchID.grid(row=2,column=0,columnspan=2,sticky="w")

    ent_searchName = tk.Entry(frame_form_search,width=30,font=fontLabel)
    ent_searchName.grid(row=4,column=0,columnspan=2,sticky="w")

    ent_searchAccuracy = tk.Entry(frame_form_search,width=30,font=fontLabel)
    ent_searchAccuracy.grid(row=6,column=0,columnspan=2,sticky="w")

    # Create Buttons
    btn_queryModel = tk.Button(frame_form_search,text="Search",command=lambda: modelSearch(frame_results_search,lbl_numResults,ent_searchID.get(),ent_searchName.get(),ent_searchAccuracy.get()),font=fontButton,width=15,height=2,bg="#D5DBDB")
    btn_queryModel.grid(row=7,column=0,pady=10,sticky="w")

    btn_deleteModel = tk.Button(frame_form_search,text="Delete",command=lambda: modelDelete(frame_results_search,lbl_numResults,ent_searchID.get()),font=fontButton,width=15,height=2,bg="#D5DBDB")
    btn_deleteModel.grid(row=7,column=1,pady=10,sticky="w")

# ------------------------ CREATE LEADERBOARD PAGE ------------------------

def pageFile():
    # Initialise Create Model Page
    winFile = tk.Toplevel()
    winFile.title("NetTest - Model Read From File")
    winFile.iconbitmap("FeatherflyCircleLogoIcon.ico")
    winFile.geometry("1200x600")
    winFile.columnconfigure(0, weight=1)

    # Create Frames
    frame_results = tk.Frame(winFile, width=1200,height=200) 
    frame_results.grid(row=0,column=0,padx=10,pady=10,sticky="nsew")
    frame_results.columnconfigure(0, weight=1)

    # Create Labels
    lbl_accuracy = tk.Label(frame_results,text="Enter Exact Model Accuracy:",font=fontLabel)
    lbl_accuracy.grid(row=1,column=0,sticky="w")

    lbl_numResults = tk.Label(frame_results,text="",font=fontLabel)
    lbl_numResults.grid(row=4,column=0,columnspan=2,sticky="w")

    # Create Entries

    ent_accuracy = tk.Entry(frame_results,font=fontLabel)
    ent_accuracy.grid(row=2,column=0,sticky="w")

    # Create Buttons

    btn_searchFile = tk.Button(frame_results, text="Search", command=lambda: searchFile(frame_results,lbl_numResults,ent_accuracy.get()),font=fontButton)
    btn_searchFile.grid(row=3,column=0,sticky="w")


    
    
# ------------------------- FONTS -----------------------

fontButton = font.Font(family='Code Bold',size=10)
fontLabel = font.Font(family="Franklin Gothic Medium",size=12)
fontLabelLarge = font.Font(family="Franklin Gothic Medium",size=18)

# ------------------------- BUTTON COMMANDS -----------------------

def trainModel(window,frame,isValidation,numHidden,epochs,mini_batch_size,eta):
    import mnist_loader
    try:
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
        training_data = list(training_data)
        print("MNIST Data Loaded")
    except:
        print("Error loading MNIST Data")

    # Instantiate object with ([numInputNeurons,numHidden,numOutputNeurons])
    net = network.Network([784, numHidden, 10])

    # Train network
    if isValidation:
        net.SGD(training_data, epochs, mini_batch_size, eta, test_data=validation_data)
        print("Using Validation Data")
    else:
        net.SGD(training_data, epochs, mini_batch_size, eta, test_data=test_data)
        print("Using Test Data")

    # Find the maximum accuracy
    net.findMax()

    # Variables
    isToFile = IntVar()

    # Display Results UI

    lbl_maxAccuracy = tk.Label(frame,text="Maximum Accuracy:",font=fontLabel)
    lbl_maxAccuracy.grid(row=3,column=0,pady=10)
    lbl_maxAccuracyValue = tk.Label(frame,text=str(net.getMaxAccuracy()) + "%",font=fontLabelLarge)
    lbl_maxAccuracyValue.grid(row=3,column=1)

    cbtn_toFile = tk.Checkbutton(frame,text="Write to File",variable=isToFile, font=fontLabel)
    cbtn_toFile.grid(row=4,column=0)

    lbl_name = tk.Label(frame,text="Enter name of model:",font=fontLabel)
    lbl_name.grid(row=5,column=0)

    ent_name = tk.Entry(frame,font=fontLabel)
    ent_name.grid(row=5,column=1)

    btn_saveModel = tk.Button(frame, text="Save", command=lambda:saveModel(frame,net,lbl_msgSave,ent_name.get(),isToFile.get()),font=fontButton, width=10,height=3,bg="#4dc425")    
    btn_saveModel.grid(row=5,column=2,pady=10,padx=10)

    lbl_msgSave = tk.Label(frame, text="", font=fontLabel)
    lbl_msgSave.grid(row=4,column=1)

    # Plot the graph in the window
    fig = Figure(figsize=(5, 3.8), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot([i for i in range(len(net.getAccuracy()))],net.getAccuracy())
    ax.set_ylabel('accuracy (%)')
    ax.set_xlabel('Number of Epochs (from 0)')


    canvas = FigureCanvasTkAgg(fig, master=frame)  # A tk.DrawingArea.
    canvas.draw()

    # pack_toolbar=False will make it easier to use a layout manager later on.
    toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
    toolbar.update()

    canvas.get_tk_widget().grid(row=1,column=0,columnspan=3)
    toolbar.grid(row=2,column=0,columnspan=3)

    # # Plot accuracy graph
    # net.plotAccuracy()

def saveModel(frame,net,label,modelName,isToFile):

    invalidChars = set(string.punctuation.replace("_", ""))
    if any(char in invalidChars for char in modelName):
        label['text'] = "ERR: No Punctuation"
        return

    net.writeModel(modelName)
    if isToFile:
        print("Writing to File")
        net.outputResults(modelName)

    label['text'] = "Model has been saved"


def modelSearch(frm,lbl,ID,name,accuracy):

    net = network.Network([0, 0, 0])

    if accuracy !="":
        try:
            acc = float(accuracy)
        except:
            lbl['text'] = "ERR: Accuracy must be a float"
            return
        if acc <0 or acc>100:
            lbl['text'] = "ERR: Accuracy is between 0 and 100"
            return
        else:
            rows,rowCount = net.getModel(ID,name,accuracy.strip())
    else:
        rows,rowCount = net.getModel(ID,name,accuracy)

    # Create Treeview
    tv = ttk.Treeview(frm, columns=(1,2,3,4,5,6,7,8,9,10), show="headings",height="16")
    tv.grid(row=1,column=0,sticky="nsew")

    # Set column widths
    tv.column(1,minwidth=30,width=60)
    for i in range(2,11):
        tv.column(i,minwidth=50,width=120)

    # Create Treeview headings
    tv.heading(1,text="Model ID")
    tv.heading(2,text="Model Name")
    tv.heading(3,text="Num Hidden Neurons")
    tv.heading(4,text="Number of Epochs")
    tv.heading(5,text="Mini-Batch Size")
    tv.heading(6,text="Learning Rate")
    tv.heading(7,text="Current Accuracy")
    tv.heading(8,text="Maximum Accuracy")
    tv.heading(9,text="Maximum Accuracy Epoch")
    tv.heading(10,text="All Accuracies")


    for row in rows:
        tv.insert('','end',values=row)

    lbl['text'] = "Number of Results: " + str(rowCount)

    #modelID, modelName, numHidden, epochs, mini_batch_size, eta, currentAccuracy, maxAccuracy, maxAccuracyEpoch, accuracyArray

def modelDelete(frm,lbl,ID):
    net = network.Network([0, 0, 0])
    deleted = net.deleteModel(ID)

    if deleted==1:
        lbl['text'] = "Record ID " +str(ID) + " Deleted"
    else:
        lbl['text'] = "No record with ID " +str(ID)

def searchFile(frm,lbl,accuracy):
    net = network.Network([0, 0, 0])
    
    savedNetworks = net.readFileData()
    sortedNetworks = net.insertionSort(savedNetworks)

    # print(savedNetworks)
    # print(sortedNetworks)

    # Create Treeview
    tv = ttk.Treeview(frm, columns=(1,2,3,4,5,6,7,8,9,10), show="headings",height="16")
    tv.grid(row=5,column=0,sticky="nsew")

    # Set column widths
    for i in range(1,10):
        tv.column(i,minwidth=50,width=120)

    # Create Treeview headings
    tv.heading(1,text="Model Name")
    tv.heading(2,text="Num Hidden Neurons")
    tv.heading(3,text="Number of Epochs")
    tv.heading(4,text="Mini-Batch Size")
    tv.heading(5,text="Learning Rate")
    tv.heading(6,text="Current Accuracy")
    tv.heading(7,text="Maximum Accuracy")
    tv.heading(8,text="Maximum Accuracy Epoch")
    tv.heading(9,text="All Accuracies")
    
    # If the accuracy is not none
    if accuracy !="":
        try:
            accuracy = float(accuracy)
        except:
            lbl['text'] = "ERR: Accuracy must be a float"
            return
        if accuracy <0 or accuracy>100:
            lbl['text'] = "ERR: Accuracy is between 0 and 100"
            return
        else:
            result = net.binarySearch(sortedNetworks,accuracy)
            if result == -1:
                lbl['text'] = "Number of Results: 0"
            else:
                    tv.insert('','end',values=result)
                    lbl['text'] = "Model Found"


        
    else:
        for record in sortedNetworks:
            tv.insert('','end',values=record)

        lbl['text'] = "Number of Results: " + str(len(sortedNetworks))

def writeDatabaseFile():
    net = network.Network([0, 0, 0])
    net.databaseToFile()
    lbl_msg_success['text'] = "Database written to file"

# ----------------------- CREATE BUTTONS -------------------------------



# Create Button Frame
frame_btn_root = tk.Frame(root, width=800,height=200) 
frame_btn_root.grid(row=2,column=0,padx=30,pady=50,sticky="nsew")

# Display Message
lbl_msg_success = tk.Label(frame_btn_root,text="",width=30,font=fontLabel)
lbl_msg_success.grid(row=3,column=0,columnspan=3)

# Create Button Widgets
btn_createModel = tk.Button(frame_btn_root, text="Create a New Model", command=pageCreateModel, font=fontButton, width=26,height=4,bg="#D5DBDB")
btn_createModel.grid(row=1,column=0,padx=(0,40),sticky="nsew")

btn_searchModel = tk.Button(frame_btn_root, text="Search Models", command=pageSearchModel, font=fontButton, width=26,height=4,bg="#D5DBDB")
btn_searchModel.grid(row=1,column=1,padx=(0,40),sticky="nsew")

btn_File = tk.Button(frame_btn_root, text="Read from File", font=fontButton, command=pageFile, width=26,height=4,bg="#D5DBDB")
btn_File.grid(row=1,column=2,sticky="nsew")

btn_databaseToFile = tk.Button(frame_btn_root, text="Write Database to File", command=writeDatabaseFile, font=fontButton, width=26,height=4,bg="#D5DBDB")
btn_databaseToFile.grid(row=2,column=0,padx=(0,40),pady=(10,0),sticky="nsew")



root.mainloop()

