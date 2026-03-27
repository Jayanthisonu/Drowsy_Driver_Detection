import tkinter as tk
from tkinter import messagebox
import pickle
import os
import cv2
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance
from pygame import mixer

# -------------------- SOUND SETUP --------------------
mixer.init()
mixer.music.load("music.wav")

# -------------------- EAR FUNCTION --------------------
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# -------------------- LOGIN FUNCTION --------------------
def validate_login():
    username = username_entry.get()
    password = password_entry.get()

    if os.path.exists("users.pkl"):
        with open("users.pkl", "rb") as f:
            users = pickle.load(f)
    else:
        users = {}

    if username in users and users[username] == password:
        show_welcome_screen(username)
    else:
        messagebox.showerror("Login Failed", "Invalid username or password")

# -------------------- SIGNUP FUNCTION --------------------
def signup():

    def save_user():
        username = new_username_entry.get()
        password = new_password_entry.get()
        confirm = confirm_password_entry.get()

        if password != confirm:
            messagebox.showerror("Error", "Passwords do not match!")
            return

        if os.path.exists("users.pkl"):
            with open("users.pkl", "rb") as f:
                users = pickle.load(f)
        else:
            users = {}

        if username in users:
            messagebox.showerror("Error", "Username already exists!")
        else:
            users[username] = password
            with open("users.pkl", "wb") as f:
                pickle.dump(users, f)

            messagebox.showinfo("Success", "User Registered Successfully!")
            signup_window.destroy()

    signup_window = tk.Toplevel(root)
    signup_window.title("Create Account")
    signup_window.geometry("400x350")
    signup_window.configure(bg="#2b2b3c")

    tk.Label(signup_window, text="Create Account",
             font=("Helvetica", 18, "bold"),
             bg="#2b2b3c", fg="white").pack(pady=20)

    tk.Label(signup_window, text="Username",
             bg="#2b2b3c", fg="white").pack()
    new_username_entry = tk.Entry(signup_window, width=25)
    new_username_entry.pack(pady=5)

    tk.Label(signup_window, text="Password",
             bg="#2b2b3c", fg="white").pack()
    new_password_entry = tk.Entry(signup_window, show="*", width=25)
    new_password_entry.pack(pady=5)

    tk.Label(signup_window, text="Confirm Password",
             bg="#2b2b3c", fg="white").pack()
    confirm_password_entry = tk.Entry(signup_window, show="*", width=25)
    confirm_password_entry.pack(pady=5)

    tk.Button(signup_window, text="Register",
              bg="#4CAF50", fg="white",
              command=save_user).pack(pady=15)

# -------------------- WELCOME SCREEN --------------------
def show_welcome_screen(username):

    root.withdraw()

    welcome_window = tk.Toplevel()
    welcome_window.title("Welcome")
    welcome_window.geometry("600x400")
    welcome_window.configure(bg="#1e1e2f")
    welcome_window.resizable(False, False)

    frame = tk.Frame(welcome_window, bg="#2b2b3c", padx=40, pady=40)
    frame.place(relx=0.5, rely=0.5, anchor="center")

    tk.Label(frame,
             text="Login Successful",
             font=("Helvetica", 22, "bold"),
             fg="#4CAF50",
             bg="#2b2b3c").pack(pady=10)

    tk.Label(frame,
             text=f"Welcome, {username}",
             font=("Helvetica", 18),
             fg="white",
             bg="#2b2b3c").pack(pady=15)

    tk.Button(frame,
              text="Start Drowsiness Detection",
              font=("Helvetica", 14, "bold"),
              bg="#2196F3",
              fg="white",
              width=25,
              command=lambda: start_detection(welcome_window)
              ).pack(pady=15)

    tk.Button(frame,
              text="Logout",
              font=("Helvetica", 12),
              bg="#f44336",
              fg="white",
              width=15,
              command=lambda: logout(welcome_window)
              ).pack(pady=10)

def start_detection(window):
    window.destroy()
    run_detection_system()
    root.deiconify()

def logout(window):
    window.destroy()
    root.deiconify()

# -------------------- DROWSINESS DETECTION --------------------
def run_detection_system():

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

    THRESH = 0.25
    FRAME_CHECK = 20

    cap = cv2.VideoCapture(0)
    flag = 0
    alarm_on = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detector(gray, 0)

        for subject in subjects:
            shape = predictor(gray, subject)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < THRESH:
                flag += 1
                if flag >= FRAME_CHECK:
                    cv2.putText(frame, "DROWSINESS ALERT!",
                                (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2)
                    if not alarm_on:
                        mixer.music.play(-1)
                        alarm_on = True
            else:
                flag = 0
                if alarm_on:
                    mixer.music.stop()
                    alarm_on = False

        cv2.imshow("Driver Drowsiness Detection System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    mixer.music.stop()

# -------------------- MAIN LOGIN WINDOW --------------------
root = tk.Tk()
root.title("Driver Drowsiness Detection System")
root.geometry("600x500")
root.configure(bg="#1e1e2f")
root.resizable(False, False)

main_frame = tk.Frame(root, bg="#2b2b3c", padx=40, pady=40)
main_frame.place(relx=0.5, rely=0.5, anchor="center")

tk.Label(main_frame,
         text="Driver Login System",
         font=("Helvetica", 22, "bold"),
         fg="white",
         bg="#2b2b3c").pack(pady=20)

tk.Label(main_frame, text="Username",
         font=("Helvetica", 14),
         fg="white", bg="#2b2b3c").pack(anchor="w")

username_entry = tk.Entry(main_frame, font=("Helvetica", 14), width=25)
username_entry.pack(pady=10)

tk.Label(main_frame, text="Password",
         font=("Helvetica", 14),
         fg="white", bg="#2b2b3c").pack(anchor="w")

password_entry = tk.Entry(main_frame, show="*", font=("Helvetica", 14), width=25)
password_entry.pack(pady=10)

tk.Button(main_frame,
          text="Login",
          font=("Helvetica", 14, "bold"),
          bg="#4CAF50",
          fg="white",
          width=20,
          command=validate_login).pack(pady=15)

tk.Button(main_frame,
          text="Create Account",
          font=("Helvetica", 12),
          bg="#2196F3",
          fg="white",
          width=20,
          command=signup).pack()

root.mainloop()