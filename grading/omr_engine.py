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

def enhance_image(image):
    """
    Apply CLAHE and other contrast enhancements to handle shadows/lighting.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced

def find_document_corners(image):
    """
    Find the 4 corners of the document in the image.
    Tries multiple strategies for robust detection on mobile.
    Returns (corners, debug_image)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Strategy 1: Canny Edges
    edged = cv2.Canny(blurred, 75, 200)
    
    strategies = [
        ("Canny", edged),
        ("Otsu", cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]),
        ("Adaptive", cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
    ]
    
    best_approx = None
    
    for name, processed in strategies:
        cnts = cv2.findContours(processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        
        if len(cnts) > 0:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                
                if len(approx) == 4 and cv2.contourArea(c) > (image.shape[0] * image.shape[1] * 0.2):
                    return approx.reshape(4, 2), edged
                
                # Keep the largest 4-point contour even if it doesn't meet the area threshold yet
                if len(approx) == 4 and (best_approx is None or cv2.contourArea(c) > cv2.contourArea(best_approx)):
                    best_approx = approx
                    
    if best_approx is not None:
        return best_approx.reshape(4, 2), edged
                
    return None, edged

def pre_process_image(image):
    """
    Basic preprocessing pipeline
    """
    return enhance_image(image)

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
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    questionCnts = []
    for c in cnts:
        (x, y, wa, ha) = cv2.boundingRect(c)
        ar = wa / float(ha)
        if wa >= 15 and ha >= 15 and ar >= 0.7 and ar <= 1.3:
            questionCnts.append(c)
            
    if not questionCnts:
        return {}, []
    
    # 1. Sort all found bubbles by Y Coordinate
    questionCnts = sorted(questionCnts, key=lambda c: cv2.boundingRect(c)[1])
    
    # 2. Cluster into Rows
    rows = []
    if questionCnts:
        current_row = [questionCnts[0]]
        for c in questionCnts[1:]:
            prev_y = cv2.boundingRect(current_row[-1])[1]
            curr_y = cv2.boundingRect(c)[1]
            if abs(curr_y - prev_y) < 20: # Same row threshold
                current_row.append(c)
            else:
                # Sort previous row left-to-right and add
                rows.append(sorted(current_row, key=lambda c: cv2.boundingRect(c)[0]))
                current_row = [c]
        rows.append(sorted(current_row, key=lambda c: cv2.boundingRect(c)[0]))
        
    results = {}
    bubble_centers = []
    
    # 3. Process each detected row
    for q_idx, row_cnts in enumerate(rows):
        if q_idx >= num_questions: break
        
        # If we have exactly the right number of bubbles, or can infer them
        # For now, simple: find the "darkest" bubble in the row
        bubbled = None
        for (j, c) in enumerate(row_cnts):
            if j >= choices: break
            
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            
            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j, c)
                
        if bubbled and bubbled[0] > 60:
            results[q_idx] = bubbled[1]
            M = cv2.moments(bubbled[2])
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                bubble_centers.append((cX, cY))
            
    return results, bubble_centers

def process_exam(image_path, num_questions=20, mcq_choices=5):
    """
    Main entry point to process a scanned exam sheet.
    """
    image = cv2.imread(image_path)
    if image is None:
        return {"success": False, "error": "Could not read image"}
        
    image = enhance_image(image)
    
    corners, debug_img = find_document_corners(image)
    if corners is None:
        return {
            "success": False, 
            "error": "Could not find document corners. Try to align the 4 corner squares in the view.",
            "debug_image": debug_img
        }
        
    w_target = 1130
    h_target = 1600
    
    dst = np.array([
        [0, 0],
        [w_target - 1, 0],
        [w_target - 1, h_target - 1],
        [0, h_target - 1]], dtype="float32")
        
    rect = order_points(corners.reshape(4, 2))
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (w_target, h_target))
    
    def to_px(mm_x, mm_y):
        return int((mm_x / 210.0) * w_target), int((mm_y / 297.0) * h_target)
        
    # Region 1: Student ID - Adjusted for new layout (x=140, y=45)
    ix1, iy1 = to_px(135, 35) 
    ix2, iy2 = to_px(180, 130) 
    roi_id = warped[iy1:iy2, ix1:ix2]
    
    # Process ID
    # Since digits 0-9 are TOP to BOTTOM, and Cols are LEFT to RIGHT.
    # To use `get_answers_from_roi` where rows=questions and cols=choices:
    # We want Rows=Columns, Choices=Digits.
    # So we rotate 90 degrees COUNTER-clockwise.
    # Top (Row 0) becomes Left. Bottom (Row 9) becomes Right.
    # Left (Col 0) becomes Bottom. Right (Col 2) becomes Top.
    # Wait, 90 CCW:
    # (x, y) -> (-y, x). 
    # Left (small x) -> Bottom (small y). Right (large x) -> Top (large y). 
    # Top (small y) -> Left (small x). Bottom (large y) -> Right (large x).
    # This means:
    # New Rows (Top to Bottom) = Original Right to Left (Col 2, 1, 0).
    # New Cols (Left to Right) = Original Top to Bottom (Row 0, 1, ..., 9).
    # Perfect! But columns are inverted. We can just flip them later or use 90 CCW + Flip.
    
    roi_id_prep = cv2.rotate(roi_id, cv2.ROTATE_90_COUNTERCLOCKWISE)
    id_results, id_centers = get_answers_from_roi(roi_id_prep, num_questions=3, choices=10)
    
    # Draw ID centers (back to warped coordinates)
    # Rotation 90 CCW: x_new = y, y_new = rot_w - x
    # Back to original ROI: x = rot_w - y_new, y = x_new
    rot_w = roi_id_prep.shape[1]
    for cx_rot, cy_rot in id_centers:
        cx_orig = rot_w - cy_rot
        cy_orig = cx_rot
        cv2.circle(warped, (cx_orig + ix1, cy_orig + iy1), 8, (0, 255, 0), -1)
        cv2.circle(warped, (cx_orig + ix1, cy_orig + iy1), 10, (255, 255, 255), 1)
    
    # RECONSTRUCT ID: Q0 is original Col 2, Q1 is Col 1, Q2 is Col 0.
    # Wait, 90 CCW maps Col 0 (Left) to Bottom, Col 2 (Right) to Top.
    # So Q0 is Col 2, Q1 is Col 1, Q2 is Col 0.
    student_id_str = ""
    for i in range(2, -1, -1): # Read in order: Q2 (Col 0), Q1 (Col 1), Q0 (Col 2)
        val = id_results.get(i, "?")
        student_id_str += str(val)
    omr_id = int(student_id_str) if student_id_str.isdigit() else None
    
    # Region 2: Answers - DYNAMIC COLUMN HANDLING
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
    final_answers = {}
    all_bubble_centers = []
    
    for c in range(num_cols):
        col_x_start = x_base_start_mm + (c * col_width_mm)
        roi_ax1, roi_ay1 = to_px(col_x_start - 2, start_y_mm - 5)
        roi_ax2, roi_ay2 = to_px(col_x_start + col_width_mm + 2, 285) 
        
        roi_col = warped[roi_ay1:roi_ay2, roi_ax1:roi_ax2]
        qs_in_this_col = min(questions_per_col, num_questions - (c * questions_per_col))
        if qs_in_this_col <= 0:
            continue
            
        col_answers, col_centers = get_answers_from_roi(roi_col, num_questions=qs_in_this_col, choices=mcq_choices)
        
        for q_rel_idx, choice_idx in col_answers.items():
            abs_q_num = (c * questions_per_col) + q_rel_idx + 1
            final_answers[abs_q_num] = choice_idx
            
        # Offset centers back to warped image
        for cx, cy in col_centers:
            all_bubble_centers.append((cx + roi_ax1, cy + roi_ay1))
            
    # DRAW VISUAL BUBBLES
    for (cx, cy) in all_bubble_centers:
        cv2.circle(warped, (cx, cy), 8, (0, 255, 0), -1) # Solid green circle
        cv2.circle(warped, (cx, cy), 10, (255, 255, 255), 1) # White outline
            
    return {
        "success": True,
        "warped_image": warped,
        "debug_image": debug_img,
        "omr_id": omr_id,
        "answers": final_answers
    }


