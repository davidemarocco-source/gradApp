import streamlit as st
import db_manager
import pandas as pd
import io

st.set_page_config(page_title="Manage Classes", page_icon="🏫")

st.title("🏫 Manage Classes & Students")

tab1, tab2, tab3 = st.tabs(["Create Class", "Add Students (Manual)", "Import Students (CSV)"])

# ----------------- CREATE CLASS -----------------
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
        for cls_id, cls_name in classes:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"**{cls_name}**")
            with col2:
                if st.button("🗑️ Delete", key=f"del_cls_{cls_id}"):
                    st.session_state[f"confirm_delete_cls_{cls_id}"] = True
            
            if st.session_state.get(f"confirm_delete_cls_{cls_id}"):
                st.warning(f"Are you sure you want to delete '{cls_name}'? This will delete all its students, exams, and results!")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Yes, Delete Everything", key=f"force_del_cls_{cls_id}"):
                        db_manager.delete_class(cls_id)
                        del st.session_state[f"confirm_delete_cls_{cls_id}"]
                        st.success(f"Class '{cls_name}' deleted.")
                        st.rerun()
                with c2:
                    if st.button("Cancel", key=f"cancel_del_cls_{cls_id}"):
                        del st.session_state[f"confirm_delete_cls_{cls_id}"]
                        st.rerun()
            st.divider()
    else:
        st.info("No classes found.")

# ----------------- MANUAL ADD -----------------
with tab2:
    st.header("Add Single Student")
    
    classes = db_manager.get_all_classes()
    if not classes:
        st.warning("Please create a class first.")
    else:
        class_options = {c[1]: c[0] for c in classes}
        selected_class_name = st.selectbox("Select Class", list(class_options.keys()), key="man_sel")
        selected_class_id = class_options[selected_class_name]
        
        with st.form("add_student_form"):
            col1, col2 = st.columns(2)
            with col1:
                student_name = st.text_input("Student Name")
            with col2:
                educational_id = st.text_input("Student ID (e.g. M2100...)")
                
            submitted = st.form_submit_button("Add Student")
            if submitted:
                if student_name and educational_id:
                    omr_id = db_manager.add_student(student_name, educational_id, selected_class_id)
                    if omr_id:
                        st.success(f"Added {student_name}. OMR ID assigned: **{omr_id}**")
                    else:
                        st.error("Error adding student. ID might be duplicate.")
                else:
                    st.warning("Please fill all fields.")
        
        # Show table
        st.divider()
        st.subheader(f"Students in {selected_class_name}")
        students = db_manager.get_students_by_class(selected_class_id)
        if students:
            df_students = pd.DataFrame(students, columns=["ID", "Name", "Edu ID", "OMR ID"])
            st.dataframe(df_students, hide_index=True)
        else:
            st.info("No students in this class.")

# ----------------- CSV IMPORT -----------------
with tab3:
    st.header("Bulk Import from CSV")
    
    classes = db_manager.get_all_classes()
    if not classes:
        st.warning("Please create a class first.")
    else:
        # Re-use logic or selectbox
        class_options_csv = {c[1]: c[0] for c in classes}
        selected_class_name_csv = st.selectbox("Select Target Class", list(class_options_csv.keys()), key="csv_sel")
        selected_class_id_csv = class_options_csv[selected_class_name_csv]
        
        st.info("Upload a CSV file with headers: `Name`, `ID`. The system will auto-assign OMR IDs.")
        
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview:")
                st.dataframe(df.head())
                
                # Check for Italian headers: ID, COGNOME, NOME
                # Or English fallbacks
                
                col_map = {c.lower(): c for c in df.columns}
                
                # Logic:
                # ID -> Educational ID
                # COGNOME + NOME -> Name
                
                id_col = None
                if "id" in col_map: 
                    id_col = col_map["id"]
                
                cognome_col = None
                if "cognome" in col_map:
                    cognome_col = col_map["cognome"]
                    
                nome_col = None
                if "nome" in col_map:
                    nome_col = col_map["nome"]
                
                # Check if we have standard Name/ID as fallback
                name_col = None
                if "name" in col_map: name_col = col_map["name"]
                
                valid = False
                if id_col:
                    if cognome_col and nome_col:
                        valid = True
                        st.success(f"Detected columns: {id_col}, {cognome_col}, {nome_col}")
                    elif name_col:
                        valid = True
                        st.success(f"Detected columns: {id_col}, {name_col}")
                
                if not valid:
                    st.error("CSV must contain columns: 'ID', 'COGNOME', 'NOME' (or 'ID', 'Name')")
                else:
                    if st.button("Import Students"):
                        count = 0
                        errors = 0
                        progress = st.progress(0)
                        
                        total = len(df)
                        for i, row in df.iterrows():
                            eid = str(row[id_col])
                            
                            if cognome_col and nome_col:
                                c = str(row[cognome_col])
                                n = str(row[nome_col])
                                nm = f"{c} {n}"
                            else:
                                nm = str(row[name_col])
                            
                            res = db_manager.add_student(nm, eid, selected_class_id_csv)
                            if res:
                                count += 1
                            else:
                                errors += 1
                            progress.progress((i + 1) / total)
                            
                        st.success(f"Imported {count} students. {errors} duplicates/errors.")
                        # Show updated list
                        students = db_manager.get_students_by_class(selected_class_id_csv)
                        if students:
                            df_students = pd.DataFrame(students, columns=["ID", "Name", "Edu ID", "OMR ID"])
                            st.dataframe(df_students, hide_index=True)
                            
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")
