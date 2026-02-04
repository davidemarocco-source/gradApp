import streamlit as st
from fpdf import FPDF
import base64

st.set_page_config(page_title="Generate Sheet", page_icon="🖨️")
st.title("🖨️ Answer Sheet Generator")

def create_sheet(num_questions=20, exam_name="Exam"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Page setup
    width = 210
    height = 297
    margin = 10
    
    # 1. Fiducial Markers (Corner Squares)
    # These are crucial for the CV algorithm to find the document.
    pdf.set_fill_color(0, 0, 0)
    marker_size = 10
    # Top-Left
    pdf.rect(margin, margin, marker_size, marker_size, 'F')
    # Top-Right
    pdf.rect(width - margin - marker_size, margin, marker_size, marker_size, 'F')
    # Bottom-Left
    pdf.rect(margin, height - margin - marker_size, marker_size, marker_size, 'F')
    # Bottom-Right
    pdf.rect(width - margin - marker_size, height - margin - marker_size, marker_size, 'F')
    
    # 2. Header
    pdf.set_xy(margin + 15, margin)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(100, 10, exam_name, ln=1)
    
    pdf.set_font("Arial", size=10)
    pdf.set_xy(margin + 15, margin + 8)
    pdf.cell(100, 10, "Name: ______________________  Date: _________", ln=1)
    
    # 3. Student ID Grid (Top Right)
    # 5 digits for Roll ID
    id_start_x = 130
    id_start_y = 25
    pdf.set_xy(id_start_x, id_start_y - 5)
    pdf.cell(50, 5, "Student Roll ID", ln=1, align='C')
    
    for row in range(10): # 0-9
        pdf.set_xy(id_start_x, id_start_y + (row * 5))
        pdf.cell(5, 5, str(row), align='R')
        for col in range(5): # 5 digits
            # Bubble
            bx = id_start_x + 10 + (col * 8)
            by = id_start_y + (row * 5) + 1
            pdf.ellipse(bx, by, 3.5, 3.5)
            
    # 4. Answers Grid
    # We'll do 2 columns
    start_y = 80
    col1_x = 30
    col2_x = 110
    
    pdf.set_font("Arial", size=10)
    
    for q in range(1, num_questions + 1):
        if q <= num_questions // 2:
            x = col1_x
            y = start_y + ((q-1) * 7)
        else:
            x = col2_x
            y = start_y + ((q - 1 - (num_questions//2)) * 7)
            
        pdf.set_xy(x, y)
        pdf.cell(10, 5, f"{q}.", align='R')
        
        # Bubbles A-E
        options = ['A', 'B', 'C', 'D', 'E']
        for i, opt in enumerate(options):
            bx = x + 15 + (i * 8)
            by = y + 1
            pdf.ellipse(bx, by, 3.5, 3.5)
            pdf.text(bx - 1, by + 1, opt) # Optional: text inside or small text below? Zipgrade has empty bubbles usually.
            # Let's put small letter inside
            # pdf.set_font("Arial", size=6)
            # pdf.text(bx-1, by+1, opt)
            # pdf.set_font("Arial", size=10)

    # 5. Numeric Section (Example placeholder)
    # If we need mixed types, the sheet generation gets complex.
    # For now, let's assume standard MCQ sheet.
    
    return pdf

exam_title = st.text_input("Exam Name for Header", "Midterm Exam")
num_q = st.number_input("Number of Questions", 10, 50, 20)

if st.button("Generate PDF"):
    pdf = create_sheet(num_q, exam_title)
    
    # Save to buffer
    pdf_output = pdf.output(dest='S').encode('latin-1')
    b64 = base64.b64encode(pdf_output).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="answer_sheet.pdf">Download PDF</a>'
    st.markdown(href, unsafe_allow_html=True)
    st.success("PDF Generated!")
