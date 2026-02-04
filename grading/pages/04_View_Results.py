import streamlit as st
import db_manager
import pandas as pd

st.set_page_config(page_title="Results", page_icon="📊")
st.title("📊 Exam Results")

# Select Exam
classes = db_manager.get_all_classes()
if not classes:
    st.warning("No data.")
    st.stop()
class_map = {c[1]: c[0] for c in classes}
selected_class = st.selectbox("Class", list(class_map.keys()))

exams = db_manager.get_exams_by_class(class_map[selected_class])
if not exams:
    st.info("No exams for this class.")
    st.stop()

exam_map = {e[1]: e[0] for e in exams} # Name -> ID. Note: duplicate names possible, ideally use ID in label
exam_opts = {f"{e[1]} ({e[2]})": e[0] for e in exams}
selected_exam_label = st.selectbox("Exam", list(exam_opts.keys()))
selected_exam_id = exam_opts[selected_exam_label]

results = db_manager.get_results_by_exam(selected_exam_id)
# student_id, name, roll_id, score

if results:
    df = pd.DataFrame(results, columns=["Student ID", "Name", "Roll ID", "Score"])
    
    # Calculate stats
    avg = df["Score"].mean()
    st.metric("Average Score", f"{avg:.2f}")
    
    st.dataframe(df, hide_index=True)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='grades.csv',
        mime='text/csv',
    )
else:
    st.info("No results graded yet.")
