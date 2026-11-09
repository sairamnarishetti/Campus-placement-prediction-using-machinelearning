from tkinter import messagebox
from tkinter import *
from tkinter import filedialog
import tkinter
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn import svm
from sklearn.metrics import accuracy_score

# -------------------- GUI SETUP --------------------
main = tkinter.Tk()
main.title("Campus Placement Prediction Using Machine Learning")
main.geometry("1500x900")
main.config(bg="#40c4b7")  # teal background

# -------------------- GLOBAL VARIABLES --------------------
filename = None
df = None

X = Y = None
X_train = X_test = None
y_train = y_test = None

classifier = None
scaler = None
feature_columns = None

# -------------------- FUNCTIONS --------------------

def upload():
    global filename, df
    filename = filedialog.askopenfilename()

    if not filename:
        return

    df = pd.read_csv(filename)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    text.delete('1.0', END)
    text.insert(END, "Dataset Loaded Successfully\n")
    text.insert(END, f"Dataset Shape : {df.shape}\n")


def splitdataset(): 
    global df, X, Y, X_train, X_test, y_train, y_test, feature_columns

    if df is None:
        messagebox.showerror("Error", "Please upload dataset first!")
        return

    X = df.drop(columns=['status'])
    Y = df['status']

    feature_columns = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=21
    )

    text.delete('1.0', END)
    text.insert(END, "Dataset Split Completed\n")
    text.insert(END, f"Training samples: {len(X_train)}\n")
    text.insert(END, f"Testing samples: {len(X_test)}\n")


def sc():
    global X_train, X_test, scaler

    if X_train is None:
        messagebox.showerror("Error", "Please split dataset first!")
        return

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    text.insert(END, "\nStandard Scaling Applied\n")


def cal_accuracy(y_test, y_pred, name): 
    acc = accuracy_score(y_test, y_pred) * 100
    text.insert(END, f"{name} Accuracy : {acc:.2f}%\n\n")
    return acc  


def logisticRegression():
    global classifier

    if X_train is None:
        messagebox.showerror("Error", "Please split dataset first!")
        return

    text.delete('1.0', END)

    cls = LogisticRegression(max_iter=1000)
    cls.fit(X_train, y_train)

    y_pred = cls.predict(X_test)
    cal_accuracy(y_test, y_pred, "Logistic Regression")

    classifier = cls


def SVM():
    global classifier

    if X_train is None:
        messagebox.showerror("Error", "Please split dataset first!")
        return

    text.delete('1.0', END)

    cls = svm.SVC(kernel='linear')
    cls.fit(X_train, y_train)

    y_pred = cls.predict(X_test)
    cal_accuracy(y_test, y_pred, "SVM")

    classifier = cls


def dt():
    global classifier

    if X_train is None:
        messagebox.showerror("Error", "Please split dataset first!")
        return

    text.delete('1.0', END)

    cls = DecisionTreeClassifier(criterion='entropy')
    cls.fit(X_train, y_train)

    y_pred = cls.predict(X_test)
    cal_accuracy(y_test, y_pred, "Decision Tree")

    classifier = cls


def predictPerformance():
    global classifier, feature_columns, scaler

    if classifier is None:
        messagebox.showerror("Error", "Train a model first!")
        return

    filename = filedialog.askopenfilename()
    if not filename:
        return

    test = pd.read_csv(filename)
    test = test.loc[:, ~test.columns.str.contains('^Unnamed')]

    if not all(col in test.columns for col in feature_columns):
        messagebox.showerror("Error", "Test data columns mismatch!")
        return

    records = test[feature_columns]

    if scaler is not None:
        records = scaler.transform(records)

    predictions = classifier.predict(records)

    text.delete('1.0', END)
    text.insert(END, "Prediction Results:\n\n")

    for val in predictions:
        if str(val) == '0':
            text.insert(END, "Not Placed\n")
        else:
            text.insert(END, "Placed\n")
            

            
            



# -------------------- UI DESIGN --------------------




# Title
title = Label(
    main,
    text="Campus Placement Prediction & Analysis using Machine Learning",
    bg="lightgray",
    fg="black",
    font=("Times New Roman", 22, "bold")
)

title.place(x=0, y=0, relwidth=1)
# Text Area
frame = Frame(main)
frame.place(x=100, y=120)

text = Text(frame, height=22, width=150, font=("Times New Roman", 12))
scroll = Scrollbar(frame, command=text.yview)
text.configure(yscrollcommand=scroll.set)

text.pack(side=LEFT)
scroll.pack(side=RIGHT, fill=Y)

# -------------------- BUTTON STYLE --------------------
btn_font = ("Times New Roman", 13, "bold")
btn_style = {
    "font": btn_font,
    "bg": "lightgray",
    "relief": RAISED,
    "bd": 3
}

# Row 1
btn_frame1 = Frame(main, bg="#40c4b7")
btn_frame1.place(x=250, y=550)

Button(btn_frame1, text="Upload Dataset", width=20, command=upload, **btn_style).grid(row=0, column=0, padx=30, pady=10)
Button(btn_frame1, text="Split Dataset", width=20, command=splitdataset, **btn_style).grid(row=0, column=1, padx=30)
Button(btn_frame1, text="Standard Scaler", width=20, command=sc, **btn_style).grid(row=0, column=2, padx=30)

# Row 2
btn_frame2 = Frame(main, bg="#40c4b7")
btn_frame2.place(x=200, y=650)

Button(btn_frame2, text="Logistic Regression", width=22, command=logisticRegression, **btn_style).grid(row=0, column=0, padx=20, pady=10)
Button(btn_frame2, text="SVM Algorithm", width=20, command=SVM, **btn_style).grid(row=0, column=1, padx=20)
Button(btn_frame2, text="Decision Tree", width=20, command=dt, **btn_style).grid(row=0, column=2, padx=20)
Button(btn_frame2, text="Predict Placement (Best Model)", width=28, command=predictPerformance, **btn_style).grid(row=0, column=3, padx=20)

# -------------------- RUN --------------------
main.mainloop()