import cv2
import face_recognition
import numpy as np
from datetime import datetime, timedelta
import pymongo

# Step 1: Connect to the MongoDB database
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["system_facial"]
collection = db["stagaire"]

# Step 2: Load face encodings from MongoDB
def load_face_encodings_from_db():
    encoded_faces = []
    names = []
    for person in collection.find():
        encoded_faces.append(np.array(person["face_encoding"], dtype=np.float64))
        names.append(person)
    return encoded_faces, names

def get_students_with_current_session():
    inscriptions_collection = db['inscriptions']
    emploi_collection = db['emploi']
    stagaire_collection = db['stagaire']

    current_time = datetime.now()

    # ki9ilib 3la l7issa fhad lw9ita 
    ongoing_sessions = emploi_collection.find({
        'start': {'$lte': current_time},
        'end': {'$gte': current_time}
    })

    # ila kanit chi 7issa 
    if ongoing_sessions.count() > 0:
        ongoing_sessions_list = list(ongoing_sessions)
        # groups limsjlin f l7issa 
        ongoing_group_ids = [session['id_group'] for session in ongoing_sessions_list]

        # ki9ilib 3la drari limsijlin f groub . bi3dima kan li9a lgroub lighadi i9ra
        students_in_sessions = inscriptions_collection.find({'id_group': {'$in': ongoing_group_ids}})
        
        # information drari msijlin 
        student_details = []
        for student_in_session in students_in_sessions:
            student_id = student_in_session['id_stagaire']
            student = stagaire_collection.find_one({'_id': student_id})
            if student:
                id_inscription = student_in_session['_id']
                id_emploi = emploi_collection.find_one({'id_group': student_in_session['id_group']})['_id']
                student_details.append({
                    'id': student['_id'],
                    'nom': student.get('nom', ''),
                    'prenom': student.get('prenom', ''),
                    'id_inscription': id_inscription,
                    'id_emploi': id_emploi,
                    'start_time': ongoing_sessions_list[0]['start'],
                    'end_time': ongoing_sessions_list[0]['end']
                })

        return student_details
    else:
        return []

# code  Cree   function  comparee face encodings dyal drari limsijlin 
def compare_face_encodings(known_encodings, face_encoding):
    matches = face_recognition.compare_faces(known_encodings, face_encoding)
    if True in matches:
        return True, matches.index(True)
    else:
        return False, None

# Cree  function lighadi t7it data bi3dima scan wjih f MongoDB 
def mark_attendance_in_db(nom, prenom, id_inscription, id_emploi, status):
    now = datetime.now()
    time = now.strftime('%I:%M:%S:%p')
    date = now.strftime('%d-%B-%Y')
    attendance_record = {
        "nom": nom,
        "prenom": prenom,
        "id_inscription": id_inscription,
        "id_emploi": id_emploi,
        "date": date,
        "time": time,
        "status": status
    }
    if status == "present":
        db["presence"].insert_one(attendance_record)
    elif status == "absent":
        db["abscent"].insert_one(attendance_record)
    elif status == "late":
        db["retard"].insert_one(attendance_record)

# code dyal camera kati9ra scan wjoh 
cap = None
for i in range(3):  # hna ki7awil yakhod 3 dyal indix scan wjih 
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        break

if cap is None or not cap.isOpened():
    print("Error: Unable to open webcam.")
else:
    known_encodings, known_names = load_face_encodings_from_db()
    current_students = get_students_with_current_session()

    if len(current_students) > 0:
        print("Students in Current Session:")
        for student in current_students:
            print(student['nom'], student['prenom'])

        # ki7sib lwi9t 
        class_end_time = current_students[0]['end_time']

        marked_attendance = set()  # kib9a kisijil 
        all_student_names = {f"{student['prenom']} {student['nom']}".lower() for student in current_students}

        while datetime.now() < class_end_time:
            success, img = cap.read()
            imgS = cv2.resize(img, (0, 0), None, 0.5, 0.5)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            faces_in_frame = face_recognition.face_locations(imgS, number_of_times_to_upsample=2)
            encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
            
            for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
                match, match_index = compare_face_encodings(known_encodings, encode_face)
                if match:
                    nom = known_names[match_index]["nom"].upper().lower()
                    prenom = known_names[match_index]["prenom"].upper().lower()
                    student_name = f"{prenom} {nom}"
                    if student_name in all_student_names:
                        if student_name not in marked_attendance:
                            for student_in_session in current_students:
                                if student_name == f"{student_in_session['prenom']} {student_in_session['nom']}".lower():
                                    start_time = student_in_session['start_time']
                                    end_time = student_in_session['end_time']
                                    time_elapsed = datetime.now() - start_time
                                    time_late = datetime.now() - end_time
                                    if time_elapsed <= timedelta(minutes=15):
                                        status = "present"
                                    elif start_time + timedelta(minutes=15) < datetime.now() <= end_time:
                                        status = "late"
                                    else:
                                         status = "absent"
                                    mark_attendance_in_db(nom, prenom, student_in_session['id_inscription'], student_in_session['id_emploi'], status)
                                    marked_attendance.add(student_name)
                        # kibiyin smiya 
                        y1, x2, y2, x1 = faceloc
                        y1, x2, y2, x1 = y1*2, x2*2, y2*2, x1*2
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, f"{prenom} {nom}", (x1+6, y2-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                else:
                    x1, y1, x2, y2 = faceloc
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, "Unknown", (x1+6, y2-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('webcam', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # kay9iyid abscence drari li no  detected
        for student_in_session in current_students:
            if f"{student_in_session['prenom']} {student_in_session['nom']}".lower() not in marked_attendance:
                mark_attendance_in_db(student_in_session['nom'], student_in_session['prenom'], student_in_session['id_inscription'], student_in_session['id_emploi'], "absent")

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("No ongoing sessions at the moment.")
