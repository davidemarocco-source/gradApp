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
        
    # Region 1: Student ID - Adjusted for new layout (x=140, y=45)
    # id_start_x = 140, id_start_y = 45. Width: 3 cols * 10 + gap. Height: 10 rows * 8.
    ix1, iy1 = to_px(135, 35) # Slightly wider/higher to be safe
    ix2, iy2 = to_px(180, 130) 
    roi_id = warped[iy1:iy2, ix1:ix2]
    
    # Process ID (Digits are rows, so rotate to process as questions)
    roi_id_rotated = cv2.rotate(roi_id, cv2.ROTATE_90_CLOCKWISE)
    id_results = get_answers_from_roi(roi_id_rotated, num_questions=3, choices=10)
    
    student_id_str = ""
    for i in range(3):
        val = id_results.get(i, "?")
        student_id_str += str(val)
    omr_id = int(student_id_str) if student_id_str.isdigit() else None
    
    # Region 2: Answers - DYNAMIC COLUMN HANDLING
    # Match the logic in 05_Sheet_Generator.py
    if num_questions <= 20:
        num_cols = 1
        col_width_mm = 80
    elif num_questions <= 40:
        num_cols = 2
        col_width_mm = 75
    else:
        num_cols = 3
        col_width_mm = 60
        
    questions_per_col = (num_questions + num_cols - 1) // num_cols
    grid_width_mm = num_cols * col_width_mm
    x_base_start_mm = (210 - grid_width_mm) / 2
    
    start_y_mm = 135
    row_height_mm = 11
    
    final_answers = {}
    
    for c in range(num_cols):
        # Calculate ROI for each column
        col_x_start = x_base_start_mm + (c * col_width_mm)
        # Give some padding to the ROI
        roi_ax1, roi_ay1 = to_px(col_x_start - 2, start_y_mm - 5)
        roi_ax2, roi_ay2 = to_px(col_x_start + col_width_mm + 2, 285) # down to bottom
        
        roi_col = warped[roi_ay1:roi_ay2, roi_ax1:roi_ax2]
        
        # Calculate how many questions are in this specific column
        qs_in_this_col = min(questions_per_col, num_questions - (c * questions_per_col))
        if qs_in_this_col <= 0:
            continue
            
        col_answers = get_answers_from_roi(roi_col, num_questions=qs_in_this_col, choices=5)
        
        for q_rel_idx, choice_idx in col_answers.items():
            abs_q_num = (c * questions_per_col) + q_rel_idx + 1
            final_answers[abs_q_num] = choice_idx
            
    return {
        "success": True,
        "warped_image": warped,
        "omr_id": omr_id,
        "answers": final_answers
    }


