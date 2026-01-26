import re

def format_ocr_text(text: str) -> str:
    """
    Formats OCR text by:
    1. Removing leading label text (e.g., "Address:", "Name:", etc.) if detected at the start
    2. Removing extra spaces (keeping only single spaces)
    3. Adding space between number-letter and letter-number transitions
    4. Ensuring space after commas and other punctuation
    
    Args:
        text: Raw OCR text to format
        
    Returns:
        Formatted and cleaned text
    """
    # First, collapse multiple spaces into single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading label text before the first colon ONLY if:
    # 1. The colon appears within the first 30 characters (indicating it's a label)
    # 2. The text before the colon doesn't contain typical address components (numbers, slashes, commas)
    if ':' in text:
        colon_index = text.find(':')
        # Only remove if colon is near the start (within first 30 chars)
        if colon_index <= 30:
            text_before_colon = text[:colon_index]
            # Check if the text before colon looks like a label
            # Labels typically don't contain numbers, slashes, or commas
            if not any(char in text_before_colon for char in ['/', ',', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
                # This looks like a label, remove it
                text = text[colon_index + 1:]
                text = text.lstrip()
    
    # Add space after commas if missing (e.g., "A/24,Road-1" -> "A/24, Road-1")
    text = re.sub(r',(?!\s)', r', ', text)
    
    # Add space between digit-letter transitions (e.g., "1Link" -> "1 Link")
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    
    # Add space between letter-digit transitions (e.g., "No17" -> "No 17")
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
    
    # Collapse any double spaces that might have been created
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing spaces
    text = text.strip()
    
    return text
