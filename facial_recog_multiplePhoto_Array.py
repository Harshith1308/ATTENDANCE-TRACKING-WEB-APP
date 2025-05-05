import os
import cv2
import face_recognition
import numpy as np

# Load and encode known student faces from a folder
def load_known_faces(folder_path):
    known_face_encodings = []
    known_face_ids = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            id_name = os.path.splitext(filename)[0]
            id, _ = id_name.split("_")  # Use only roll number

            image_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_ids.append(id)

    return known_face_encodings, known_face_ids

# Process images and mark attendance
def process_attendance(image_paths, save_directory, known_face_encodings, known_face_ids):
    present_students = set()  # Use set to avoid duplicates

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for img_path in image_paths:
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                rollno = known_face_ids[best_match_index]
                present_students.add(rollno)

            # Optional: draw bounding box
            cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

        save_path = os.path.join(save_directory, os.path.basename(img_path))
        cv2.imwrite(save_path, image_bgr)

    # Internally store sorted roll numbers (no return)
    global sorted_present_rollnos
    sorted_present_rollnos = sorted(present_students)

# === Main Execution ===

# Path to folder with known student images
data_folder = "C:/Users/Harsh/PycharmProjects/Facial_Recognition/facial_recog/students"

# Input images for attendance
image_paths = ["IMG-20250401-WA0001.jpg","IMG-20250401-WA0002.jpg","IMG-20250401-WA0003.jpg","IMG-20250401-WA0005.jpg","IMG-20250401-WA0007.jpg","IMG-20250401-WA0009.jpg","IMG-20250401-WA0011.jpg","IMG-20250401-WA0013.jpg","IMG-20250401-WA0015.jpg","IMG-20250401-WA0017.jpg","IMG-20250401-WA0019.jpg","IMG-20250401-WA0033.jpg","group_2_scatter.jpg"]
# Where to save processed images
save_directory = "C:/Users/Harsh/PycharmProjects/Facial_Recognition/facial_recog/ALL_PICS"

# Load known student encodings
known_face_encodings, known_face_ids = load_known_faces(data_folder)

# Process attendance
process_attendance(image_paths, save_directory, known_face_encodings, known_face_ids)

# Now `sorted_present_rollnos` contains the sorted roll numbers
# You can use it later like:
# print(sorted_present_rollnos)  <-- optional, not included per your request
