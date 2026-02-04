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
    # End: (130 + 30, 25 + 60) # roughly covers the grid (50mm wide was 5 cols, 3 cols ~ 30mm)
    # Actually, let's keep it generous to capture margin.
    x1, y1 = to_px(125, 20)
    x2, y2 = to_px(160, 85)
    roi_id = warped[y1:y2, x1:x2]
    
    # Process ID
    # This is a grid of 10 rows (0-9) x 3 cols (digits)
    # The `get_answers_from_roi` sorts contours top-to-bottom (rows).
    # Since our grid is Vertical digits (row 0 is value 0 for all columns),
    # Row 0 contains bubbles for '0' for the 100s, 10s, 1s place.
    # Basically:
    # Row 0: Bubble for 0 (Col 1), Bubble for 0 (Col 2), Bubble for 0 (Col 3)
    # We want to find which row is bubbled for each column.
    
    # Let's rotate the ROI 90 degrees? No, the bubbling logic is standard:
    # Question = Row. Answer = Column.
    # Here: Row = Digit Value (0-9). Column = Digit Position (100s, 10s, 1s).
    # So `get_answers_from_roi` will return: {RowIdx (0-9): ColIdx (0-2)}.
    # This means: "For Value 5 (Row 5), the student bubbled Column 1 (Tens place)".
    # This format is valid but we need to reconstruct the full number.
    # E.g. ID = 105.
    # Row 0: Bubbled Col 1 (10s place has 0). Result: {0: 1}
    # Row 1: Bubbled Col 0 (100s place has 1). Result: {1: 0}
    # Row 5: Bubbled Col 2 (1s place has 5). Result: {5: 2}
    # This relies on the student bubbling only ONE bubble per row? NO!
    # A student might bubble '0' in Col 2 and '0' in Col 3 (ID 500).
    # `get_answers_from_roi` assumes ONE answer per row. It won't work if multiple columns are bubbled in the same row.
    # (e.g. ID 111 -> Row 1 has 3 bubbles. Logic will pick one).
    
    # CONCLUSION: Standard multichoice logic (Row=Question) DOES NOT apply to this grid orientation easily
    # UNLESS we treat COLUMNS as Questions.
    # If we rotate the image 90 degrees, then:
    # Columns become Rows.
    # Col 1 (100s) -> Becomes Row 1. Bubbles 0..9 are choices A..J.
    # This is exactly what we need!
    
    roi_id_rotated = cv2.rotate(roi_id, cv2.ROTATE_90_CLOCKWISE)
    # Now: 
    # Top-to-Bottom are the *Original Left-to-Right Columns* (100s, 10s, 1s).
    # Left-to-Right are the *Original Top-to-Bottom Rows* (0..9).
    
    # We have 3 "Questions" (digits). Each has 10 Choices (0-9).
    id_results = get_answers_from_roi(roi_id_rotated, num_questions=3, choices=10)
    
    # Returns {QuestionIdx (0-2): AnswerIdx (0-9)}
    # Q0 -> 100s place. Answer 1 -> Value 1.
    # This is perfect.
    
    student_id_str = ""
    for i in range(3):
        val = id_results.get(i, "?")
        student_id_str += str(val)
            
    omr_id = int(student_id_str) if student_id_str.isdigit() else None
    
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
        "omr_id": omr_id,
        "answers": final_answers
    }


