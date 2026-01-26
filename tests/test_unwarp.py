import os
import cv2
import numpy as np
from paddleocr import TextImageUnwarping

def extend_image(image, padding_h=50, padding_w=50):
    """
    Extends the image by adding a white border.
    """
    h, w = image.shape[:2]
    new_h = h + 2 * padding_h
    new_w = w + 2 * padding_w
    extended_img = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255
    extended_img[padding_h:padding_h+h, padding_w:padding_w+w] = image
    return extended_img

def main():
    # Define paths
    input_image_path = os.path.abspath("samples/testing/t2.png")
    output_dir = os.path.abspath("output")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    if not os.path.exists(input_image_path):
        print(f"Error: Input image not found at {input_image_path}")
        return

    print(f"Processing image: {input_image_path}\n")

    # Step 1: Read image
    print("Step 1: Reading image...")
    img = cv2.imread(input_image_path)
    if img is None:
        print("Error: Failed to read image.")
        return
    print(f"   Image loaded: {img.shape}\n")

    # Step 2: Extend image
    print("Step 2: Extending image...")
    extended_img = extend_image(img, padding_h=1, padding_w=20)
    extended_img_path = os.path.join(output_dir, "unwarp_test_extended.jpg")
    cv2.imwrite(extended_img_path, extended_img)
    print(f"   Extended image saved: {extended_img_path}")
    print(f"   Extended image shape: {extended_img.shape}\n")
    
    # Step 3: Text Image Rectification (Unwarping)
    print("Step 3: Text Image Rectification (Unwarping)...")
    try:
        # Initialize the model
        print("   Initializing TextImageUnwarping model...")
        unwarp_model = TextImageUnwarping(model_name="UVDoc")
        
        # Run prediction
        print(f"   Running prediction on: {extended_img_path}")
        unwarp_result = unwarp_model.predict(extended_img_path, batch_size=1)
        
        # Step 4: Save the unwarped image
        print("\nStep 4: Saving unwarped image...")
        for res in unwarp_result:
            # Print the result
            print("   Result structure:")
            res.print()
            
            # Save to image using the documented method
            print(f"\n   Saving to directory: {output_dir}")
            res.save_to_img(save_path=output_dir)
            
            # Also save JSON for inspection
            json_path = os.path.join(output_dir, "unwarp_test_result.json")
            res.save_to_json(save_path=json_path)
            print(f"   Saved JSON result to: {json_path}")
            
            break
        
        print("\n" + "="*50)
        print("Unwarping complete!")
        print(f"Check the '{output_dir}' directory for output files.")
        print("="*50)
        
    except Exception as e:
        print(f"   Error during unwarping: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
