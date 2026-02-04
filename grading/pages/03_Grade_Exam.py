import streamlit as st
import db_manager
import omr_engine
import cv2
import numpy as np
import json
from PIL import Image

st.set_page_config(page_title="Grade Exam", page_icon="📸")

st.title("📸 Grade Exam")

# 1. Select Exam
exams = db_manager.get_all_classes() 
# We need to select class first? Or just list all recent exams?
# Let's list all exams with their class names.
# Need a better DB query for that, but let's iterate.
all_classes = db_manager.get_all_classes()
if not all_classes:
    st.warning("No classes/exams found.")
    st.stop()

class_map = {c[1]: c[0] for c in all_classes}
selected_class_name = st.selectbox("Select Class", list(class_map.keys()))
selected_class_id = class_map[selected_class_name]

exams = db_manager.get_exams_by_class(selected_class_id)
if not exams:
    st.warning("No exams for this class.")
    st.stop()
    
exam_opts = {f"{e[1]} ({e[2]})": e[0] for e in exams}
selected_exam_label = st.selectbox("Select Exam", list(exam_opts.keys()))
selected_exam_id = exam_opts[selected_exam_label]

# Load Exam Details (Key)
exam_details = db_manager.get_exam_details(selected_exam_id)
# id, name, class_id, date, answer_key
answer_key = json.loads(exam_details[4]) 
# key format: {"1": "A", "2": 3.5, ...} but from json it comes as strings?
# Actually my save logic was: key_data[q] = ans. q is int.
# json converts int keys to strings.
# So answer_key will be {"1": "A", "2": "B"}

# 2. Input Method
input_method = st.radio("Input Method", ["Upload Image", "Camera"])

image_file = None
if input_method == "Upload Image":
    image_file = st.file_uploader("Upload Scanned Sheet", type=['jpg', 'png', 'jpeg'])
else:
    image_file = st.camera_input("Take a picture of the sheet")

if image_file:
    # Convert to CV2
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    st.image(image, caption="Original Image", channels="BGR", use_container_width=True)
    
    if st.button("Process & Grade"):
        with st.spinner("Analyzing..."):
            # Save temp file for processing (engine takes path)
            temp_path = "temp_scan.jpg"
            cv2.imwrite(temp_path, image)
            
            # Count questions from key
            num_qs = len(answer_key)
            
            # Process
            result = omr_engine.process_exam(temp_path, num_questions=num_qs)
            
            if result["success"]:
                st.success("Processing Complete!")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(result["warped_image"], caption="Warped View", channels="BGR")
                
                with col2:
                    st.subheader("Results")
                    roll_id = result["roll_id"]
                    st.write(f"**Detected Roll ID:** {roll_id}")
                    
                    # Try to find student
                    student = db_manager.get_student_by_roll(roll_id)
                    student_id = None
                    if student:
                        st.success(f"Matched Student: **{student[1]}**")
                        student_id = student[0]
                    else:
                        st.error("Student not found in DB!")
                        # Allow manual override
                        student_list = db_manager.get_students_by_class(selected_class_id)
                        stu_opts = {s[1]: s[0] for s in student_list}
                        sel_stu = st.selectbox("Manually Select Student", list(stu_opts.keys()))
                        student_id = stu_opts[sel_stu]
                        
                    # Grading
                    student_answers = result["answers"] # {1: 0, 2: 1} (0=A, 1=B...)
                    
                    score = 0
                    total = 0
                    
                    graded_details = {}
                    
                    for q_str, proper_ans in answer_key.items():
                        q_idx = int(q_str)
                        
                        # Student answer
                        # My engine returns {1: 0}, 0 index.
                        # My key has {1: "A"}.
                        # Need map.
                        idx_to_char = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
                        
                        stu_ans_idx = student_answers.get(q_idx)
                        stu_ans_char = idx_to_char.get(stu_ans_idx, "?") if stu_ans_idx is not None else "N/A"
                        
                        is_correct = False
                        if stu_ans_char == proper_ans:
                            is_correct = True
                            score += 1
                        
                        graded_details[q_idx] = {
                            "student": stu_ans_char,
                            "correct": proper_ans,
                            "is_correct": is_correct
                        }
                        total += 1
                        
                    st.metric("Score", f"{score} / {total}")
                    
                    # Save
                    if st.button("Save Grade"):
                        # image path: usually save to disk properly with unique name
                        # for demo we skip preserving image
                        db_manager.save_result(selected_exam_id, student_id, score, graded_details, "scan.jpg")
                        st.success("Saved to Database!")
                        
            else:
                st.error(f"Failed: {result['error']}")
