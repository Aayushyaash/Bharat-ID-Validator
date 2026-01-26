import numpy as np
import cv2
from fastapi import UploadFile

async def read_image_file(file: UploadFile) -> np.ndarray:
    """
    Reads an uploaded file and converts it to an OpenCV image (numpy array).
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    """
    Rotates an image by the specified angle (0, 90, 180, 270).
    Angle is expected to be in degrees counter-clockwise.
    """
    if angle == 0:
        return image
    
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) # PaddleOCR returns 90 for "rotated 90 degrees", so we need to rotate back? 
        # Wait, usually orientation detection tells you how much it is rotated. 
        # If Paddle says "90", it usually means the text is at 90 degrees. To fix it, we rotate -90 (or 270) or generic logic.
        # However, the plan says: "Extract angle (0, 90, 180, 270). If angle != 0, call image_utils.rotate_image."
        # Let's assume the angle passed here is the *correction* angle or I need to deduce the correction.
        # PaddleOCR `cls` output usually gives the angle of the text (0, 90, 180, 270). 
        # If text is at 90, we probably want to rotate it -90 (270) to make it upright? 
        # Or if the plan implies "rotate_image" takes the DETECTED angle and corrects it.
        # Let's implement generic rotation for now.
        pass
    
    # Actually, cv2.rotate constants are:
    # ROTATE_90_CLOCKWISE
    # ROTATE_180
    # ROTATE_90_COUNTERCLOCKWISE
    
    # If the input angle is 90, I will rotate 90 clockwise? Or counter? 
    # Standard interpretation: If I want to rotate BY 90 degrees.
    
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return image

def extend_image(image: np.ndarray, padding_h: int = 1, padding_w: int = 20) -> np.ndarray:
    """
    Extends the image by adding a white border/padding.
    This improves OCR accuracy for text near edges.
    
    Args:
        image: Input image as numpy array
        padding_h: Padding height to add on top and bottom (default: 1)
        padding_w: Padding width to add on left and right (default: 20)
        
    Returns:
        Extended image with white padding
    """
    h, w = image.shape[:2]
    new_h = h + 2 * padding_h
    new_w = w + 2 * padding_w
    
    # Create white background image
    extended_img = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255
    
    # Place original image in the center
    extended_img[padding_h:padding_h+h, padding_w:padding_w+w] = image
    
    return extended_img
