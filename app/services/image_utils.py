import numpy as np
import cv2
from fastapi import UploadFile


async def read_image_file(file: UploadFile) -> np.ndarray | None:
    """
    Reads an uploaded file and converts it to an OpenCV image.
    
    Args:
        file: FastAPI UploadFile object containing the image data
        
    Returns:
        Image as numpy array in BGR format (OpenCV standard), or None if
        the file cannot be decoded as an image
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    """
    Rotates an image by the specified angle.
    
    Supports only 90-degree increments (0, 90, 180, 270) using OpenCV's
    efficient rotation functions. This is used for document orientation
    correction after detecting the current rotation.
    
    Args:
        image: Input image as numpy array
        angle: Rotation angle in degrees clockwise (0, 90, 180, or 270)
        
    Returns:
        Rotated image as numpy array, or original image if angle is 0
        or not a supported value
        
    Example:
        If document is detected as rotated 90° clockwise, pass angle=270
        to rotate it back (or equivalently, 90° counter-clockwise)
    """
    if angle == 0:
        return image
    
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
