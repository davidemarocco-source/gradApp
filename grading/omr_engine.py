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

def sample_bubble(warped_image, cx, cy, radius=8):
    """
    Check the darkness of a circular region at centered at (cx, cy).
    Returns (average_intensity, bubble_mask_visual)
    Since we are using 0=black, 255=white, lower means DARKER.
    """
    mask = np.zeros(warped_image.shape[:2], dtype="uint8")
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    
    # Get mean intensity of the grayscale version
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    mean_val = cv2.mean(gray, mask=mask)[0]
    return mean_val

def get_answers_from_roi(roi, num_questions=5, choices=5):
    """
    Deprecated: Using coordinate-based sampling instead.
    """
    return {}

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
        
    all_bubble_centers = []
    
    # --- 1. Process Student ID (3 columns of digits 0-9) ---
    # Layout matches Sheet Generator: 
    # id_start_x = 140, id_start_y = 50
    # cols = 3, rows = 10 (0-9)
    # col_spacing = 10, row_spacing = 8
    id_start_x = 140
    id_start_y = 50
    id_digits = []
    
    for c in range(3): # Col 0 (leftmost) to Col 2
        col_x = id_start_x + (c * 10)
        darkest_row = None
        min_intensity = 255
        
        for r in range(10): # Row 0 to 9
            row_y = id_start_y + (r * 8)
            px, py = to_px(col_x, row_y)
            intensity = sample_bubble(warped, px, py, radius=7)
            
            # 0=Black, 255=White. Darker = lower.
            if intensity < min_intensity:
                min_intensity = intensity
                darkest_row = r
        
        # Threshold: Paper is usually > 200. Bubbles are ~150. Filled is < 100.
        if darkest_row is not None and min_intensity < 180:
            id_digits.append(str(darkest_row))
            # Mark detected bubble
            px, py = to_px(id_start_x + (c * 10), id_start_y + (darkest_row * 8))
            all_bubble_centers.append((px, py))
        else:
            id_digits.append("?")
            
    student_id_str = "".join(id_digits)
    omr_id = int(student_id_str) if "?" not in student_id_str else None
    
    # --- 2. Process Answer Grid ---
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
    # Bubble positions in a row: 
    # x = start_x + 15 + (j * 10)
    
    final_answers = {}
    
    for c in range(num_cols):
        col_x_start = x_base_start_mm + (c * col_width_mm)
        qs_in_this_col = min(questions_per_col, num_questions - (c * questions_per_col))
        
        for q_idx in range(qs_in_this_col):
            abs_q_num = (c * questions_per_col) + q_idx + 1
            row_y = start_y_mm + (q_idx * row_height_mm)
            
            darkest_choice = None
            min_intensity = 255
            
            for j in range(mcq_choices):
                bubble_x = col_x_start + 15 + (j * 10)
                px, py = to_px(bubble_x, row_y + 2) # +2 for slight text offset
                intensity = sample_bubble(warped, px, py, radius=8)
                
                if intensity < min_intensity:
                    min_intensity = intensity
                    darkest_choice = j
            
            if darkest_choice is not None and min_intensity < 180:
                final_answers[abs_q_num] = darkest_choice
                # Mark detected bubble
                px, py = to_px(col_x_start + 15 + (darkest_choice * 10), row_y + 2)
                all_bubble_centers.append((px, py))
            
    # DRAW VISUAL BUBBLES
    for (cx, cy) in all_bubble_centers:
        cv2.circle(warped, (cx, cy), 10, (0, 255, 0), 2) # Green ring
        cv2.circle(warped, (cx, cy), 2, (0, 255, 0), -1) # Center dot
            
    return {
        "success": True,
        "warped_image": warped,
        "debug_image": debug_img,
        "omr_id": omr_id,
        "answers": final_answers
    }


