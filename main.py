import cv2
import pickle
import tkinter as tk
import mediapipe as mp
import numpy as np
from PIL import Image
from PIL import ImageTk
import speech_recognition as sr

from dict import text_map

# Load model
with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)


# Config
font = "Courier"
font_size = 44
fontButtons = (font, 12)
window_width = 800
window_height = 700
webcam_width = 800
webcam_height = 450

# Init main window
main_window = tk.Tk('main')
main_window.geometry('%dx%d+%d+%d' % (500, 200, 200, 200))
main_window.title('Main window')
sign_to_text_btn = tk.Button(main_window, text="Sign to text", font=font)
speech_to_text_btn = tk.Button(main_window, text='Speech to text', font=font)
sign_to_text_btn.pack(side='left', padx=40)
speech_to_text_btn.pack(side='right', padx=40)

# Init sign to text window
sign_to_text_window = tk.Toplevel(main_window)
sign_to_text_window.geometry('%dx%d+%d+%d' %
                             (window_width, window_height, 0, 0))
is_stream = False

# Init webcam capture using opencv
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Webcam stream
webcam_frame = tk.Frame(sign_to_text_window,
                        width=webcam_width, height=webcam_width)
webcam_frame.place(x=0, y=0)
webcam = tk.Label(webcam_frame)
webcam.pack(side='top')

# Current sign text
current_sign_widget = tk.Label(
    sign_to_text_window,
    text='Hiện tại: ',
    font=(font, 44)
)
current_sign_widget.place(x=10, y=webcam_height+20, height=font_size + 20)

# Segment sign widget
seg_text = ''
seg_widget = tk.Label(sign_to_text_window,
                      text='Văn bản :', font=(font, font_size))
seg_widget.place(x=10, y=webcam_height+font_size+40)

# Back button
back_btn = tk.Button(sign_to_text_window, text="Quay lai",
                     font=(font, 30), )
back_btn.pack(side='bottom', pady=10)


# Speech to text screen
speech_to_text_window = tk.Toplevel(main_window)
speech_to_text_window.geometry('%dx%d+%d+%d' %
                               (window_width, window_height, 400, 0))
speech_to_text_window.title('Speech to text')

record_btn = tk.Button(speech_to_text_window, text='Record', font=(font, 30))
record_btn.place(x=10, y=10)

speech_text = tk.Label(speech_to_text_window, text='', font=(font, 20))
speech_text.place(x=10, y=100)

image_box = tk.Frame(speech_to_text_window, width=450, height=450)
image_box.pack()
image_box.place(x=350, y=0)
image_label = tk.Label(image_box)
image_label.pack(side='right')


def image_processed(hand_img):
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
    img_flip = cv2.flip(img_rgb, 1)
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.7
    )

    output = hands.process(img_flip)
    hands.close()

    try:
        data = output.multi_hand_landmarks[0]
        data = str(data)
        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']
        without_garbage = []
        for i in data:
            if i not in garbage:
                without_garbage.append(i)

        clean = []

        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])
        return (clean)
    except:
        return (np.zeros([1, 63], dtype=int)[0])


def show_frame():
    if not is_stream:
        return

    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")

    frame = cv2.flip(frame, 1)

    # Predict image
    data = image_processed(frame)
    data = np.array(data)
    y_pred = svm.predict(data.reshape(-1, 63))
    result = str(y_pred[0])

    global current_sign_widget
    current_sign_widget.config(text='Hiện tại: ' + result)

    global seg_text
    if len(result) == 1 and (len(seg_text) == 0 or result != seg_text[-1]):
        seg_text += result
    seg_widget.config(text='Văn bản : '+seg_text)

    print(result)

    # Stream webcam
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    img = Image.fromarray(cv2image).resize((webcam_width, webcam_height))
    imgTk = ImageTk.PhotoImage(image=img)
    webcam.imgtk = imgTk
    webcam.configure(image=imgTk)
    webcam.after(10, show_frame)


def speech_to_text():
    speech_recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print('Speech')
        audio = speech_recognizer.listen(source)
        try:
            text = speech_recognizer.recognize_google(audio, language="vi-VI")
            print(text)
            global speech_text
            speech_text.config(text=text)

            print(text_map[str(text).lower()])

            img_path = text_map[str(text).lower()]
            img = Image.open('./assets/' + img_path).resize((450, 450), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=img)

            global image_label
            image_label.imgtk = imgtk
            image_label.configure(image=imgtk)

        except Exception as err:
            print(err)

    # return text


def open_sign_to_text_window():
    main_window.withdraw()
    speech_to_text_window.withdraw()
    global is_stream
    is_stream = True
    show_frame()
    sign_to_text_window.deiconify()


def open_speech_to_text():
    main_window.withdraw()
    sign_to_text_window.withdraw()
    speech_to_text_window.deiconify()


def on_back_main_window():
    print('Back')
    global is_stream
    is_stream = False
    sign_to_text_window.withdraw()
    speech_to_text_window.withdraw()
    main_window.deiconify()


# Speech to text
record_btn.config(command=speech_to_text)


sign_to_text_btn.config(command=open_sign_to_text_window)
speech_to_text_btn.config(command=open_speech_to_text)


sign_to_text_window.withdraw()
speech_to_text_window.withdraw()

sign_to_text_window.protocol("WM_DELETE_WINDOW", on_back_main_window)
speech_to_text_window.protocol("WM_DELETE_WINDOW", on_back_main_window)
back_btn.config(command=on_back_main_window)


speech_to_text_window.mainloop()
sign_to_text_window.mainloop()


main_window.mainloop()
