import cv2
import face_recognition
import numpy as np
import os
import time
import smtplib
import imghdr
from email.message import EmailMessage
from datetime import datetime

# -------------------- Load Known Faces -------------------- #
known_face_encodings = []
known_face_names = []
known_faces_dir = "known_faces"

for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])
        else:
            print(f"⚠️ No face found in {filename}. Skipping.")

print("✅ Known faces loaded successfully!")

# -------------------- Setup Webcam -------------------- #
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("❌ Failed to access the webcam. Please check if it's connected and not being used by another application.")
    exit()
else:
    print("📷 Webcam connected successfully!")

# Set camera resolution to 640x480 for better speed
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# -------------------- Email Settings -------------------- #
EMAIL_SENDER = "sender email"
EMAIL_PASSWORD = "your app password"  # Consider using environment variables for security
EMAIL_RECEIVER = "receiver email"

def send_email(image_path):
    """Sends an email with an unknown face image."""
    msg = EmailMessage()
    msg["Subject"] = "🚨 Unknown Face Detected!"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.set_content(f"An unknown face was detected. See the attached image: {image_path}")

    with open(image_path, "rb") as img:
        img_data = img.read()
        img_type = imghdr.what(img.name)
        msg.add_attachment(img_data, maintype="image", subtype=img_type, filename=os.path.basename(image_path))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"📧 Email sent with image {image_path}")
    except Exception as e:
        print(f"❌ Email sending failed: {e}")

# -------------------- Face Recognition Logic -------------------- #
face_names = []
last_recognition_time = {}
RECOGNITION_DELAY = 30  # seconds
frame_skip = 2  # Process every 2nd frame for speed
frame_count = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("❌ Camera error. Exiting...")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    new_face_names = []
    current_time = time.time()

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if matches else None

        if best_match_index is not None and matches[best_match_index] and face_distances[best_match_index] < 0.4:
            name = known_face_names[best_match_index]
            last_seen = last_recognition_time.get(name, 0)

            if current_time - last_seen > RECOGNITION_DELAY:
                print(f"✅ Welcome {name}!")
                os.system(f"espeak 'Welcome {name}'")
                last_recognition_time[name] = current_time
            new_face_names.append(name)
        else:
            name = "Unknown"
            new_face_names.append(name)

            if current_time - last_recognition_time.get("Unknown", 0) > RECOGNITION_DELAY:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                os.makedirs("recognized_faces", exist_ok=True)
                image_path = f"recognized_faces/unknown_{timestamp}.jpg"
                cv2.imwrite(image_path, frame)
                print(f"❗ Unknown face detected! Image saved at {image_path}")

                send_email(image_path)
                last_recognition_time["Unknown"] = current_time

    face_names = new_face_names

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
