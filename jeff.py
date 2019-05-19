

from tkinter import Tk, filedialog

import numpy as np





def select_file(file_type, file_type_description, Window_Title):
# This function asks for the user to select a file in a window and returns the parent folder path and file name
#Usage:
# parent_folder, file_name  = select_file(file_type, file_type_description, Window_Title)
#Where:
# parent folder is the path of where the selected file is
# file_name is the name of the selected file
# file_type is the file extention e.g. "*.txt"
# file_type_description is the description of the file type
# Window_title = "Select a file" pr any other appropriate description

    # Initiating Tkinter
    root = Tk()

    # Prevent a Tk window to open for Windows only
    try:
        root.widthdraw()
    except:
        # to destroy tkinter tab for Mac that otherwise appears alongside
        root.iconify()

    # To bring tab to the foreground
    root.lift()

    # Prevent a bug
    root.update()

    # open the file selection window
    f_path = filedialog.askopenfilename(filetypes=[(file_type_description,file_type)],title = Window_Title)

    # checking if the operation was cancelled
    if f_path=="":
        raise ValueError('Cancelled...')
    else:
        print(f_path)

    # completely switch off tkinter
    root.destroy()

    # Extracting the file name from the full path
    f_file = f_path.split("/")[-1]

    # Extracting the path to parent folder from the full path
    f_folder=f_path[0:len(f_path)-len(f_file)]

    return f_folder, f_file


a , b = select_file("*.txt", "Text file", "Select a file")

def read_three_columns(f_path = ""):
    if f_path = 
    f=open(f_path,'r')
    data=f.read()
    f.close()
    data=data.replace('\n\n','\n')
    data=data.split()
    #converting str to floats and splitting in x, y and z
    x=np.asarray([float(i) for i in data[0::3]])
    y=np.asarray([float(i) for i in data[1::3]])
    z=np.asarray([float(i) for i in data[2::3]])