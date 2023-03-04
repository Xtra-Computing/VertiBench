# implement a GUI window. The window contains a button and a text box.
# When the button is clicked, the text box will show the today's date and time.

import tkinter as tk
from datetime import datetime

# create a window
window = tk.Tk()
window.title('Playground')
window.geometry('300x200')

# create a button
button = tk.Button(window, text='Click me')
button.pack()

# create a text box
text = tk.Text(window, height=2)
text.pack()

# define a function to show the date and time
def show_date_time():
    date_time = datetime.now()
    text.insert('end', date_time)

# bind the button with the function
button.bind('<Button-1>', lambda e: show_date_time())

window.mainloop()
