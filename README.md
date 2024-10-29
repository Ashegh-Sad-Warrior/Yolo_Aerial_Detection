# YOLO Aerial Detection Model

This repository contains a YOLO-based model for aerial object detection. The model is trained to detect and classify various objects from aerial images, such as planes, ships, vehicles, and more. Below are the details and visualizations regarding the performance of the model.

## Model Description

- **Model Name**: YOLO Aerial Mine Detection
- **Framework**: Ultralytics YOLOv11n-obb
- **Languages**: English, Persian
- **Classes Detected**:
  - Plane (هواپیما)
  - Ship (کشتی)
  - Storage Tank (مخزن ذخیره)
  - Baseball Diamond (زمین بیسبال)
  - Tennis Court (زمین تنیس)
  - Basketball Court (زمین بسکتبال)
  - Ground Track Field (زمین دو و میدانی)
  - Harbor (بندرگاه)
  - Bridge (پل)
  - Large Vehicle (خودرو بزرگ)
  - Small Vehicle (خودرو کوچک)
  - Helicopter (هلیکوپتر)
  - Roundabout (میدان)
  - Soccer Ball Field (زمین فوتبال)
  - Swimming Pool (استخر شنا)

## Training Details

- **Dataset**: Custom aerial images annotated for object detection.
- **Metrics**: Precision, Recall, mAP@0.5, F1 Score
- **Training Environment**: Kaggle, GPU-accelerated environment
- **Optimizer**: SGD
- **Libraries Used**:
  - **Ultralytics**: YOLOv11n-obb (version 8.0.0)
  - **Gradio**: For creating the user interface (version 3.1.4)
  - **Pandas**: For data handling (version 1.3.3)
  - **Pillow**: For image manipulation (version 8.4.0)
  - **OpenCV**: For video processing (version 4.5.3)

## Evaluation Results

Below are the various evaluation results obtained during the training and testing phases of the model.

### F1-Confidence Curve
![F1-Confidence Curve](F1_curve.png)

### Precision-Confidence Curve
![Precision-Confidence Curve](P_curve.png)

### Precision-Recall Curve
![Precision-Recall Curve](PR_curve.png)

### Recall-Confidence Curve
![Recall-Confidence Curve](R_curve.png)

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### Labels Correlogram
![Labels Correlogram](labels_correlogram.jpg)

### Labels Distribution
![Labels Distribution](labels.jpg)

## How to Use

1. **Clone this repository.**
2. **Load the model using the Ultralytics YOLO library.**
3. **Use the model for object detection on aerial images.**

```python
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

# Load the YOLO model
model = YOLO('/content/best.pt')

# Predict on an image
results = model.predict(source='/content/boats.jpg', conf=0.3)
obb_results = results[0].obb  # Extract Oriented Bounding Boxes (OBB) from the results

# Load the original image using OpenCV
image = cv2.imread('/content/boats.jpg')

# Check if there are any detection results
if obb_results is not None and len(obb_results.data) > 0:
    # Iterate over each detected object
    for obb, conf, cls in zip(
        obb_results.data.cpu().numpy(), 
        obb_results.conf.cpu().numpy(), 
        obb_results.cls.cpu().numpy()
    ):
        # Extract bounding box parameters
        x_center, y_center, width, height, rotation = obb[:5]
        class_id = int(cls)
        confidence = float(conf)

        # Define the rotated rectangle (bounding box)
        rect = (
            (x_center, y_center),  # Center of the rectangle
            (width, height),       # Size of the rectangle
            rotation * 180.0 / np.pi  # Rotation angle in degrees
        )

        # Get the four corners of the rotated rectangle
        box = cv2.boxPoints(rect)
        box = np.int0(box)  # Convert coordinates to integers

        # Draw the rotated bounding box on the image
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)  # Green color with thickness 2

        # Put the class ID and confidence score near the bounding box
        cv2.putText(
            image, 
            f'Class {class_id}, Conf {confidence:.2f}', 
            (int(x_center), int(y_center)), 
            cv2.FONT_HERSHEY_SIMPLEX,  # Font type
            0.5,                        # Font scale
            (0, 255, 0),                # Text color (Green)
            1,                          # Thickness of the text
            cv2.LINE_AA                 # Line type for anti-aliasing
        )

    # Convert the image from BGR (OpenCV format) to RGB (Matplotlib format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image with bounding boxes using Matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide axis
    plt.show()
else:
    print("No objects detected.")

 ```

License

This model is open-sourced under the MIT License.
Acknowledgements

Special thanks to the Kaggle community and Hugging Face for providing tools and platforms for developing and sharing this project.

### **Key Changes and Enhancements:**

1. **Properly Closed Code Block:**
   - The Python code in the "How to Use" section is enclosed within triple backticks (\`\`\`) with `python` specified for syntax highlighting at the beginning.
   - Added triple backticks after the Python code to properly close the code block.
   
   ```markdown
   ```python
   # Your Python code here

2. **Consistent Heading Levels:**
- Ensured all headings use appropriate Markdown syntax (`#`, `##`, `###`) for better organization and readability.

3. **Bullet Points and Lists:**
- Used clear and consistent bullet points for lists, enhancing readability.

4. **Image Links:**
- Verified that image links use the correct Markdown syntax. Make sure that the images (e.g., `F1_curve.png`, `P_curve.png`) are uploaded to your repository in the correct paths.

5. **Dependencies Section (Optional but Recommended):**
- Although not present in your original README, it's highly recommended to include a `requirements.txt` file in your repository for easy installation of dependencies. Here’s how you can add it:

**Create a `requirements.txt` file with the following content:**
ultralytics==8.0.0 gradio==3.1.4 pandas==1.3.3 Pillow==8.4.0 opencv-python==4.5.3 matplotlib==3.4.3 numpy==1.21.2


**Include instructions in the README on how to install dependencies:**
```markdown
## Installation

To install the required libraries, run:

```bash
pip install -r requirements.txt


6. **Clarify Training Details:**
- Ensure that all training details accurately reflect the configuration used. For example, if you used `SGD` as the optimizer, confirm that this matches your training script.

7. **Enhance "How to Use" Instructions:**
- Consider adding more detailed instructions or examples on how to use the model, interpret results, or integrate it into larger applications.

8. **Consistency and Clarity:**
- Maintained consistent formatting throughout the README for a professional appearance.
- Ensured that all instructions are clear and easy to follow, especially for users who may not be familiar with YOLO or object detection.

### **Additional Recommendations:**

1. **Ensure All Images Are Uploaded:**
- Verify that all images referenced in the Evaluation Results section are correctly uploaded to the repository and that their paths in the Markdown match their locations.

2. **Check Syntax Highlighting:**
- Using `python` after the backticks helps in proper syntax highlighting, making the code more readable.

3. **Test the README:**
- Before finalizing, preview the README in your repository to ensure that all sections, especially the code blocks and images, render correctly.

4. **Version Control:**
- Keep track of the versions of the libraries used. This is already partially addressed in the Training Details, but ensuring consistency in environments can help avoid compatibility issues.

5. **Contact Information (Optional):**
- If you expect others to collaborate on your project, consider adding sections for contact information or contribution guidelines.

### **Final Tips:**

- **Verify All Paths:**
- Ensure that the paths to both the model (`yolo11n-obb.pt`) and the test image (`boats.jpg`) are correct. If you're deploying on Hugging Face Spaces or another platform, adjust these paths based on your repository structure.

- **Provide Additional Resources:**
- Optionally, include links to relevant documentation or resources, such as the [Ultralytics YOLO documentation](https://docs.ultralytics.com/) or [Gradio guides](https://gradio.app/), to help users understand and utilize your model effectively.

- **Handle Multiple Detections Correctly:**
- Ensure that the `counts` dictionary is initialized outside the loop to correctly count multiple detected objects.

- **Add a `requirements.txt` File:**
- Including a `requirements.txt` file helps others easily set up the necessary environment by running:
 ```bash
 pip install -r requirements.txt

By following these guidelines and ensuring that your Markdown syntax is correct, your README file will be well-structured, easy to read, and professional, making it easier for others to understand, use, and contribute to your YOLO Aerial Detection Model on Hugging Face.

If you need further assistance or have specific sections you'd like to refine, feel free to ask!

