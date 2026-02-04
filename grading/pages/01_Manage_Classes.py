import streamlit as st
import db_manager
import pandas as pd

st.set_page_config(page_title="Manage Classes", page_icon="🏫")

st.title("🏫 Manage Classes & Students")

tab1, tab2 = st.tabs(["Create Class", "Add Students"])

with tab1:
    st.header("Create New Class")
    new_class_name = st.text_input("Class Name")
    if st.button("Create Class"):
        if new_class_name:
            if db_manager.add_class(new_class_name):
                st.success(f"Class '{new_class_name}' created!")
            else:
                st.error("Class already exists.")
        else:
            st.warning("Please enter a class name.")
            
    st.divider()
    st.subheader("Existing Classes")
    classes = db_manager.get_all_classes()
    if classes:
        df_classes = pd.DataFrame(classes, columns=["ID", "Name"])
        st.dataframe(df_classes, hide_index=True)
    else:
        st.info("No classes found.")

with tab2:
    st.header("Add Students")
    
    classes = db_manager.get_all_classes()
    if not classes:
        st.warning("Please create a class first.")
    else:
        class_options = {c[1]: c[0] for c in classes}
        selected_class_name = st.selectbox("Select Class", list(class_options.keys()))
        selected_class_id = class_options[selected_class_name]
        
        with st.form("add_student_form"):
            col1, col2 = st.columns(2)
            with col1:
                student_name = st.text_input("Student Name")
            with col2:
                student_roll = st.text_input("Roll ID (Unique)")
                
            submitted = st.form_submit_button("Add Student")
            if submitted:
                if student_name and student_roll:
                    if db_manager.add_student(student_name, student_roll, selected_class_id):
                        st.success(f"Added {student_name} to {selected_class_name}")
                    else:
                        st.error("Error adding student. Roll ID might be duplicate.")
                else:
                    st.warning("Please fill all fields.")
        
        st.divider()
        st.subheader(f"Students in {selected_class_name}")
        students = db_manager.get_students_by_class(selected_class_id)
        if students:
            df_students = pd.DataFrame(students, columns=["ID", "Name", "Roll ID"])
            st.dataframe(df_students, hide_index=True)
        else:
            st.info("No students in this class.")
