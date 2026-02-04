import cv2
import numpy as np

def order_points(pts):
    """
    Rearrange coordinates to order: top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left
    rect[2] = pts[np.argmax(s)] # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-right
    rect[3] = pts[np.argmax(diff)] # Bottom-left

    return rect

def four_point_transform(image, pts):
    """
    Apply perspective transform to obtain a top-down view of the image
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width of new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute height of new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute perspective matrix and warp
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def find_document_corners(image):
    """
    Find the 4 corners of the document in the image.
    Returns None if not found, otherwise returns the 4 points.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    # Sort contours by area, descending
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        
        for c in cnts:
            # Approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # If our approximated contour has 4 points, we assume we found the document
            if len(approx) == 4:
                return approx.reshape(4, 2)
                
    return None

def pre_process_image(image):
    """
    Basic preprocessing pipeline
    """
    # Resize if too large to improve speed, keep aspect ratio? 
    # For now, let's just work with original or standard width
    return image

def sort_contours(cnts, method="left-to-right"):
    """
    Sort contours based on the provided method.
    """
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b: b[1][i], reverse=reverse))
        
    return (cnts, boundingBoxes)

def get_answers_from_roi(roi, num_questions=5, choices=5):
    """
    Process a ROI containing just the bubbles.
    Assumes the ROI is already perspective warped to just the question grid.
    Returns a dictionary of question_index -> chosen_index
    """
    # 1. Binarize
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # 2. Find contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    questionCnts = []
    
    # 3. Filter for bubbles
    # Heuristic: Aspect ratio approx 1, and size relative to image
    h, w = roi.shape[:2]
    min_w, max_w = w // (choices * 2), w // choices # approximate width of a single bubble slot
    
    for c in cnts:
        (x, y, wa, ha) = cv2.boundingRect(c)
        ar = wa / float(ha)
        
        # Adjust these heuristics based on the actual sheet design
        # For now, let's just use strict aspect ratio and some size bounds
        if wa >= 20 and ha >= 20 and ar >= 0.8 and ar <= 1.2:
            questionCnts.append(c)
            
    # 4. Sort top-to-bottom to isolate rows (questions)
    if not questionCnts:
        return {}
        
    questionCnts = sort_contours(questionCnts, method="top-to-bottom")[0]
    
    expected_bubbles = num_questions * choices
    # Debug: print(f"Found {len(questionCnts)} bubbles. Expected {expected_bubbles}")
    
    # Needs robust handling if we miss bubbles. 
    # Ideally, we should grid-slice the image rather than rely solely on contour finding if lighting is bad.
    # But for now, let's assume we found them all.
    
    results = {}
    
    # Loop over questions (rows)
    for (q, i) in enumerate(np.arange(0, len(questionCnts), choices)):
        # Get the contours for the current question
        cnts = sort_contours(questionCnts[i:i + choices])[0]
        
        bubbled = None
        
        # Determine which bubble is filled
        # We check the mask of each bubble
        for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            
            # Count non-zero pixels in the thresholded image within this mask
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            
            # Simple heuristic: the one with most pixels is the answer
            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)
                
        # Store the result (index of the choice)
        # You might want a threshold here to say "unanswered"
        if bubbled:
            results[q] = bubbled[1]
            
    return results

def process_exam(image_path, num_questions=20):
    """
    Main entry point to process a scanned exam sheet.
    Returns:
        {
            "success": bool,
            "warped_image": numpy array (or None),
            "roll_id": str,
            "answers": dict {q_idx: answer_idx},
            "error": str
        }
    """
    image = cv2.imread(image_path)
    if image is None:
        return {"success": False, "error": "Could not read image"}
        
    # 1. basic resize for consistency if needed?
    # image = cv2.resize(image, (width, height))
    
    # 2. Find Corners
    corners = find_document_corners(image)
    if corners is None:
        return {"success": False, "error": "Could not find document corners"}
        
    # 3. Warp to A4 proportions (approx 210x297)
    # Let's map to a high-res flat image, e.g., 840x1188 (multiply by 4)
    # Aspect Ratio A4 = 1.414
    # Let's use a standard height of 1600 px
    w_target = 1130 # 1600 / sqrt(2)
    h_target = 1600
    
    # We must construct destination points for the warp
    dst = np.array([
        [0, 0],
        [w_target - 1, 0],
        [w_target - 1, h_target - 1],
        [0, h_target - 1]], dtype="float32")
        
    rect = order_points(corners.reshape(4, 2))
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (w_target, h_target))
    
    # 4. Extract Regions
    # Based on PDF Layout:
    # Width=210, Height=297
    # Margins=10
    # ID Grid: X=130, Y=25. Size: ~5*8=40mm wide, 10*5=50mm high.
    # Answers: Two columns.
    
    def to_px(mm_x, mm_y):
        return int((mm_x / 210.0) * w_target), int((mm_y / 297.0) * h_target)
        
    # Region 1: Student ID
    # Start: (130, 25)
    # End: (130 + 50, 25 + 60) # roughly covers the grid
    x1, y1 = to_px(130, 25)
    x2, y2 = to_px(180, 85)
    roi_id = warped[y1:y2, x1:x2]
    
    # Process ID
    # This is a grid of 10 rows (0-9) x 5 cols (digits)
    # We read it as "Vertical" question? No, rows are numbers 0-9. 
    # Actually, usually ID grids are: Cols=Digits, Rows=0-9. 
    # You bubble '0' in col 1, '1' in col 2... 
    # My PDF generator did: Row 0..9, Col 0..4.
    # So iterating rows 0-9 means we are looking for which number is bubbled in each column?
    # Wait, my PDF generator code:
    # for row in range(10): pdf.cell(..., str(row))...
    #    for col in range(5): ...
    # This implies the grid is 10 rows high, 5 columns wide.
    # Row 0 contains bubbles for '0' for all 5 digits?
    # Yes. So to find the ID, we look at each COLUMN and see which ROW is bubbled.
    # Standard OMR function `get_answers_from_roi` scans ROWS as questions.
    # So if we simply rotate the ROI 90 degrees, we can reuse `get_answers_from_roi`?
    # Or just write a specific `process_id_grid`.
    
    id_results = get_answers_from_roi(roi_id, num_questions=10, choices=5)
    # This returns {row_idx: col_idx}. 
    # row_idx (0-9) corresponds to the digit value.
    # col_idx (0-4) corresponds to the position of the digit (10000s, 1000s, etc).
    # We want {col_idx: row_idx}.
    
    # Actually `get_answers_from_roi` scans top-to-bottom.
    # It will find 10 rows of bubbles.
    # For each row (number 0..9), it finds which column (digit place) is bubbled.
    # This means if I bubble '2' in the first column, 
    # in row index 2, column index 0 will be detected.
    
    student_id = ["?"] * 5
    for number_val, digit_pos in id_results.items():
        # number_val is the row (0-9)
        # digit_pos is the column (0-4)
        if 0 <= digit_pos < 5:
            student_id[digit_pos] = str(number_val)
            
    roll_id = "".join(student_id)
    
    # Region 2: Answers
    # Col 1: X=30, Y=80. 
    # Col 2: X=110, Y=80.
    # Height per Q = 7mm.
    # Let's just grab the whole bottom area.
    # Split into 2 ROIs.
    
    # ROI Answer Col 1
    # X: 30 to 80 (approx 50mm width)
    # Y: 80 to bottom margin (say 280)
    ax1, ay1 = to_px(30, 80)
    ax2, ay2 = to_px(90, 200) # Covers first 10-15 qs
    roi_ans1 = warped[ay1:ay2, ax1:ax2]
    
    # ROI Answer Col 2
    bx1, by1 = to_px(110, 80)
    bx2, by2 = to_px(170, 200)
    roi_ans2 = warped[by1:by2, bx1:bx2]
    
    # Process
    # Assume 10 questions per column for now (based on PDF default 20 Qs)
    # In a real app we'd calculate ROI height based on num_questions
    
    q_per_col = num_questions // 2
    
    ans1 = get_answers_from_roi(roi_ans1, num_questions=q_per_col, choices=5)
    ans2 = get_answers_from_roi(roi_ans2, num_questions=num_questions - q_per_col, choices=5)
    
    final_answers = {}
    
    for q_idx, choice_idx in ans1.items():
        # q_idx is 0-based relative to ROI
        final_answers[q_idx + 1] = choice_idx 
        
    for q_idx, choice_idx in ans2.items():
        final_answers[q_idx + 1 + q_per_col] = choice_idx
        
    return {
        "success": True,
        "warped_image": warped,
        "roll_id": roll_id,
        "answers": final_answers
    }


