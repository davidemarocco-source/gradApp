import streamlit as st
from fpdf import FPDF
import base64
import db_manager
import json

st.set_page_config(page_title="Generate Sheet", page_icon="🖨️")
st.title("🖨️ Answer Sheet Generator")

# --- Exam Selection Logic ---
classes = db_manager.get_all_classes()
class_options = {c[1]: c[0] for c in classes}

# Check if we came from Manage Exams
selected_exam_id = st.session_state.get('selected_exam_id')

with st.sidebar:
    st.header("Load Existing Exam")
    if not classes:
        st.warning("No classes found. Create one first.")
    else:
        # Default class and exam if coming from redirect
        default_class_idx = 0
        if 'selected_exam_class_id' in st.session_state:
            for i, (name, id) in enumerate(class_options.items()):
                if id == st.session_state['selected_exam_class_id']:
                    default_class_idx = i
                    break
        
        sel_class_name = st.selectbox("Select Class", list(class_options.keys()), index=default_class_idx)
        class_id = class_options[sel_class_name]
        
        exams = db_manager.get_exams_by_class(class_id)
        if exams:
            exam_options = {e[1]: e[0] for e in exams}
            
            default_exam_idx = 0
            if selected_exam_id:
                for i, (name, id) in enumerate(exam_options.items()):
                    if id == selected_exam_id:
                        default_exam_idx = i
                        break
            
            sel_exam_name = st.selectbox("Select Exam", list(exam_options.keys()), index=default_exam_idx)
            
            if st.button("Load Exam Details"):
                exam_details = db_manager.get_exam_details(exam_options[sel_exam_name])
                if exam_details:
                    answer_key = json.loads(exam_details[4])
                    st.session_state['gen_exam_name'] = exam_details[1]
                    st.session_state['gen_num_q'] = len(answer_key)
                    st.session_state['gen_mcq_choices'] = exam_details[5]
                    # Clear redirect state once handled
                    if 'selected_exam_id' in st.session_state:
                        del st.session_state['selected_exam_id']
                        del st.session_state['selected_exam_class_id']
                    st.rerun()
        else:
            st.info("No exams found for this class.")

st.divider()

def create_sheet(num_questions=20, exam_name="Exam", mcq_choices=5):
    pdf = FPDF()
    pdf.add_page()
    
    # Use standard fonts (Helvetica is safe and looks professional)
    pdf.set_font("Helvetica", size=12)
    
    # Page setup
    width = 210
    height = 297
    margin = 15
    
    # 1. Fiducial Markers (Corner Squares) - Increased to 10mm for better detection
    pdf.set_fill_color(0, 0, 0)
    marker_size = 10
    # Top-Left
    pdf.rect(margin, margin, marker_size, marker_size, 'F')
    # Top-Right
    pdf.rect(width - margin - marker_size, margin, marker_size, marker_size, 'F')
    # Bottom-Left
    pdf.rect(margin, height - margin - marker_size, marker_size, marker_size, 'F')
    # Bottom-Right
    pdf.rect(width - margin - marker_size, height - margin - marker_size, marker_size, marker_size, 'F')
    
    # 2. Header
    header_x = margin + 15
    pdf.set_xy(header_x, margin)
    pdf.set_font("Helvetica", 'B', 20)
    pdf.cell(100, 10, exam_name.upper(), ln=1)
    
    pdf.set_font("Helvetica", size=12)
    pdf.set_xy(header_x, margin + 12)
    pdf.cell(100, 10, f"NAME: {'_'*40}    DATE: {'_'*15}", ln=1)
    
    # 3. Student ID Grid (Top Right)
    id_start_x = 140
    id_start_y = 45
    pdf.set_xy(id_start_x, id_start_y - 8)
    pdf.set_font("Helvetica", 'B', 10)
    pdf.cell(55, 5, "STUDENT ID", ln=1, align='C')
    
    bubble_r = 5.5 # Increased from 4.5
    bubble_gap_x = 10 # Increased from 8
    bubble_gap_y = 8 # Increased from 6
    
    for col in range(3):
        col_x = id_start_x + (col * bubble_gap_x) + 12
        for row in range(10):
            bx = col_x
            by = id_start_y + (row * bubble_gap_y)
            pdf.ellipse(bx, by, bubble_r, bubble_r)
            pdf.set_font("Helvetica", size=8)
            pdf.set_xy(bx, by)
            pdf.cell(bubble_r, bubble_r, str(row), align='C')
            
    pdf.set_xy(id_start_x, id_start_y + 82)
    pdf.set_font("Helvetica", 'I', 8)
    pdf.cell(55, 5, "Riempi i cerchi completamente", ln=1, align='C')
            
    # 4. Answers Grid
    start_y = 135 # Moved down more to give ID grid room
    # Determine columns based on num_questions
    if num_questions <= 20:
        num_cols = 1
        col_width = 80
    elif num_questions <= 40:
        num_cols = 2
        col_width = 75
    else:
        num_cols = 3
        col_width = 60
    
    questions_per_col = (num_questions + num_cols - 1) // num_cols
    bubble_size = 6.5 # Increased from 5
    bubble_spacing = 9 # Increased from 7.5
    row_height = 11 # Increased from 9
    
    for q in range(1, num_questions + 1):
        col_idx = (q - 1) // questions_per_col
        q_idx = (q - 1) % questions_per_col
        
        grid_width = num_cols * col_width
        x_base = (width - grid_width) / 2 + (col_idx * col_width)
        y = start_y + (q_idx * row_height)
        
        # Question Number
        pdf.set_font("Helvetica", 'B', 11)
        pdf.set_xy(x_base, y)
        pdf.cell(12, row_height, f"{q}.", align='R')
        
        # Bubbles (Dynamic Choices)
        options = ['A', 'B', 'C', 'D', 'E'][:mcq_choices]
        pdf.set_font("Helvetica", size=8)
        for i, opt in enumerate(options):
            bx = x_base + 15 + (i * bubble_spacing)
            by = y + (row_height - bubble_size) / 2
            pdf.ellipse(bx, by, bubble_size, bubble_size)
            pdf.set_xy(bx, by)
            pdf.cell(bubble_size, bubble_size, opt, align='C')
            
    return pdf

# Use session state for inputs if available
default_name = st.session_state.get('gen_exam_name', "Midterm Exam")
default_num_q = st.session_state.get('gen_num_q', 20)
default_choices = st.session_state.get('gen_mcq_choices', 5)

exam_title = st.text_input("Exam Name for Header", value=default_name)
num_q = st.number_input("Number of Questions", 1, 100, value=default_num_q)
mcq_choices = st.number_input("MCQ Choices (2-5)", 2, 5, value=default_choices)

if st.button("Generate PDF"):
    pdf = create_sheet(num_q, exam_title, mcq_choices)
    
    # Save to buffer
    pdf_output = pdf.output(dest='S').encode('latin-1')
    b64 = base64.b64encode(pdf_output).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="answer_sheet.pdf">Download PDF</a>'
    st.markdown(href, unsafe_allow_html=True)
    st.success("PDF Generated!")
