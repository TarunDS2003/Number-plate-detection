import cv2
import numpy as np
import pytesseract
import os
from datetime import datetime

# Set the path to the Tesseract executable (update this path if necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Create a directory to store detected number plate images
output_dir = 'detected_number_plates'
os.makedirs(output_dir, exist_ok=True)

# Create a text file to store the extracted number plate texts
output_text_file = 'number_plates.txt'

# Set to keep track of already detected number plates
detected_number_plates = set()

def detect_number_plate(frame):
    global detected_number_plates
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)

    # Apply morphological operations to enhance edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # Find contours based on edges detected
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area and aspect ratio to find number plate contours
    number_plate_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # Check if the contour has 4 vertices (rectangle shape)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # Further filter based on aspect ratio and area
            if 2.5 < aspect_ratio < 5.5 and cv2.contourArea(contour) > 1000:
                number_plate_contour = approx
                break

    if number_plate_contour is not None:
        # Draw the contour on the original image
        cv2.drawContours(frame, [number_plate_contour], -1, (0, 255, 0), 3)

        # Create a mask for the number plate
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [number_plate_contour], 0, 255, -1)
        new_image = cv2.bitwise_and(frame, frame, mask=mask)

        # Crop the bounding rectangle of the number plate
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        cropped_color = frame[topx:bottomx+1, topy:bottomy+1]

        # Convert the cropped image to grayscale for OCR
        cropped_gray = cv2.cvtColor(cropped_color, cv2.COLOR_BGR2GRAY)

        # Use OCR to read the number plate
        number_plate_text = pytesseract.image_to_string(cropped_gray, config='--psm 8')
        number_plate_text = ''.join(e for e in number_plate_text if e.isalnum() and (e.isupper() or e.isdigit()))  # Filter out non-uppercase letters and non-digits
        if number_plate_text:
            # Save the cropped image to the output directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(output_dir, f'{number_plate_text}_{timestamp}.png')
            cv2.imwrite(output_path, cropped_color)
            print(f'Number plate detected: {number_plate_text}')
            print(f'Saved to: {output_path}')

            # Append the number plate text to the output text file
            with open(output_text_file, 'a') as f:
                f.write(f'{number_plate_text}\n')
            
            # Add the detected number plate to the set
            detected_number_plates.add(number_plate_text)

        return frame, cropped_color
    else:
        return frame, None

def process_stored_images():
    for filename in os.listdir(output_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(output_dir, filename)
            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            number_plate_text = pytesseract.image_to_string(gray_image, config='--psm 8')
            number_plate_text = ''.join(e for e in number_plate_text if e.isalnum() and (e.isupper() or e.isdigit()))  # Filter out non-uppercase letters and non-digits
            if number_plate_text:
                print(f'Processing {filename}: {number_plate_text}')
                with open(output_text_file, 'a') as f:
                    f.write(f'{filename}: {number_plate_text}\n')

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, cropped = detect_number_plate(frame)

        cv2.imshow('Video Feed', frame)
        if cropped is not None:
            cv2.imshow('Detected Number Plate', cropped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Process stored images to extract text
    process_stored_images()

if __name__ == "__main__":
    main()