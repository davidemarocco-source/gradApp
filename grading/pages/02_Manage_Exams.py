import streamlit as st
import db_manager
import datetime
import json

st.set_page_config(page_title="Manage Exams", page_icon="📝")

st.title("📝 Header Exams & Keys")

classes = db_manager.get_all_classes()
if not classes:
    st.warning("Please create a class first.")
    st.stop()
    
class_options = {c[1]: c[0] for c in classes}

with st.expander("Create New Exam", expanded=True):
    with st.form("create_exam_form"):
        exam_name = st.text_input("Exam Name")
        selected_class = st.selectbox("Class", list(class_options.keys()))
        exam_date = st.date_input("Date", datetime.date.today())
        
        st.subheader("Answer Key")
        num_questions = st.number_input("Number of Questions", min_value=1, max_value=100, value=10)
        
        # We can't dynamically add form rows inside a form based on input in the same form easily without rerun.
        # So we submit the basic info first? Or we use state.
        # Let's use a two-step process: Define Meta -> Define Key.
        
        submitted = st.form_submit_button("Start Key Definition")
        
        if submitted:
            if exam_name:
                st.session_state['draft_exam'] = {
                    "name": exam_name,
                    "class_id": class_options[selected_class],
                    "date": str(exam_date),
                    "num_questions": num_questions
                }
                st.rerun()
            else:
                st.error("Exam name required")

if 'draft_exam' in st.session_state:
    st.divider()
    draft = st.session_state['draft_exam']
    st.subheader(f"Define Key for: {draft['name']}")
    
    with st.form("key_form"):
        key_data = {}
        cols = st.columns(5)
        
        for q in range(1, draft['num_questions'] + 1):
            with cols[(q-1)%5]:
                # Options: A, B, C, D, E, or Numeric Value
                # Simple approach: Text input for flexibility, or Selectbox.
                # Let's use a selectbox for standard MCQ and a toggle for numeric.
                
                q_type = st.selectbox(f"Q{q} Type", ["MCQ", "Numeric"], key=f"q_type_{q}", label_visibility="collapsed")
                
                if q_type == "MCQ":
                   ans = st.selectbox(f"Q{q} Ans", ["A", "B", "C", "D", "E"], key=f"q_{q}")
                   key_data[q] = ans
                else:
                   ans = st.number_input(f"Q{q} Val", key=f"q_{q}", step=0.1)
                   key_data[q] = ans

        if st.form_submit_button("Save Exam"):
            db_manager.create_exam(draft['name'], draft['class_id'], draft['date'], key_data)
            st.success("Exam Saved!")
            del st.session_state['draft_exam']
            st.rerun()

st.divider()
st.subheader("Existing Exams")
# List exams by class
selected_view_class = st.selectbox("Filter by Class", ["All"] + list(class_options.keys()))

if selected_view_class == "All":
    # Need a get_all_exams function or loop through classes. 
    # For simplicity, let's just show raw list if we had a generic function, 
    # but db helper is by_class. I'll just iterate if "All" or fix db helper.
    # Let's just prompt user to select class.
    st.info("Select a class to view exams")
else:
    exams = db_manager.get_exams_by_class(class_options[selected_view_class])
    if exams:
        for ex in exams:
            with st.expander(f"{ex[1]} ({ex[2]})"):
                details = db_manager.get_exam_details(ex[0])
                st.json(json.loads(details[4]))
