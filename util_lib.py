'''
Created on May 11, 2023


'''

import tkinter as tk
from tkinter import filedialog


def select_filename_dialog(start_directory, dialog_title, multiple_files = False):
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    file_path = filedialog.askopenfilename(initialdir = start_directory, title = dialog_title, multiple = multiple_files)
    return  file_path
