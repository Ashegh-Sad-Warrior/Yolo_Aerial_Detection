# !pip install ultralytics
# !pip install gradio

import cv2
from ultralytics import YOLO
from PIL import Image
import gradio as gr
import pandas as pd
import numpy as np
import tempfile
import os

# Check if the model file exists
if os.path.exists('/content/best.pt'):
    print("Model file found.")
else:
    print("Model file not found. Please upload 'best.pt' to the Space.")

# Load your trained model
model = YOLO('best.pt')  # Adjust the model path accordingly

# Define class names in English
class_names = {
    0: 'plane',
    1: 'ship',
    2: 'storage tank',
    3: 'baseball diamond',
    4: 'tennis court',
    5: 'basketball court',
    6: 'ground track field',
    7: 'harbor',
    8: 'bridge',
    9: 'large vehicle',
    10: 'small vehicle',
    11: 'helicopter',
    12: 'roundabout',
    13: 'soccer ball field',
    14: 'swimming pool'
}

# Define colors for each class (BGR for OpenCV)
colors = {
    0: (255, 0, 0),       # Red
    1: (0, 255, 0),       # Green
    2: (0, 0, 255),       # Blue
    3: (255, 255, 0),     # Yellow
    4: (255, 0, 255),     # Magenta
    5: (0, 255, 255),     # Cyan
    6: (128, 0, 128),     # Purple
    7: (255, 165, 0),     # Orange
    8: (0, 128, 0),       # Dark Green
    9: (128, 128, 0),     # Olive
    10: (0, 255, 0),      # Light Green for class_id=10
    11: (0, 128, 128),    # Teal
    12: (0, 0, 128),      # Navy
    13: (75, 0, 130),     # Indigo
    14: (199, 21, 133)    # Medium Violet Red
}

# Function to detect objects in images
def detect_and_draw_image(input_image):
    try:
        # Convert PIL image to a NumPy array (RGB)
        input_image_np = np.array(input_image)
        print("Image converted to NumPy array.")

        # Run the model on the image using the NumPy array (RGB)
        results = model.predict(source=input_image_np, conf=0.3)
        print("Model prediction completed.")

        # Access Oriented Bounding Box (OBB) results
        if hasattr(results[0], 'obb') and results[0].obb is not None:
            obb_results = results[0].obb
            print("Accessed obb_results.")
        else:
            print("No 'obb' attribute found in results[0].")
            obb_results = None

        # Check if any detections are found
        if obb_results is None or len(obb_results.data) == 0:
            print("No objects detected.")
            df = pd.DataFrame({
                'Label': [],
                'Object Count': []
            })
            return input_image, df

        counts = {}
        # Process results and draw bounding boxes
        for obb, conf, cls in zip(obb_results.data.cpu().numpy(), obb_results.conf.cpu().numpy(), obb_results.cls.cpu().numpy()):
            x_center, y_center, width, height, rotation = obb[:5]
            class_id = int(cls)
            confidence = float(conf)

            # Draw the rotated bounding box using OpenCV
            rect = ((x_center, y_center), (width, height), rotation * 180.0 / np.pi)  # Convert radians to degrees
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)
            color = colors.get(class_id, (0, 255, 0))
            cv2.drawContours(input_image_np, [box_points], 0, color, 1)  # Set thickness to 1
            print(f"Drawn OBB for class_id {class_id} with confidence {confidence}.")

            # Draw label with appropriate position and reduced thickness
            label = class_names.get(class_id, 'unknown')
            text_position = (int(x_center), int(y_center) - int(height / 2) - 10)
            cv2.putText(input_image_np, f'{label}: {confidence:.2f}',
                        text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)  # Set font thickness to 1

            # Count detected objects
            counts[label] = counts.get(label, 0) + 1

        # Convert image to RGB for Gradio
        image_rgb = cv2.cvtColor(input_image_np, cv2.COLOR_BGR2RGB)
        output_image = Image.fromarray(image_rgb)
        print("Image converted back to RGB for Gradio.")

        # Create DataFrame to display results
        df = pd.DataFrame({
            'Label': list(counts.keys()),
            'Object Count': list(counts.values())
        })
        print("DataFrame created.")

        return output_image, df

    except Exception as e:
        print(f"Error in detect_and_draw_image: {e}")
        df = pd.DataFrame({
            'Label': [],
            'Object Count': []
        })
        return input_image, df

# Function to detect objects in videos
def detect_and_draw_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        overall_counts = {}
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            print(f"Processing frame {frame_count}")

            # Resize the frame
            frame = cv2.resize(frame, (640, 480))

            # Run the model on the frame
            results = model.predict(source=frame, conf=0.3)
            print(f"Model prediction completed for frame {frame_count}.")

            # Access Oriented Bounding Box (OBB) results
            if hasattr(results[0], 'obb') and results[0].obb is not None:
                obb_results = results[0].obb
                print("Accessed obb_results for frame.")
            else:
                print("No 'obb' attribute found in results[0] for frame.")
                obb_results = None

            if obb_results is not None and len(obb_results.data) > 0:
                for obb, conf, cls in zip(obb_results.data.cpu().numpy(), obb_results.conf.cpu().numpy(), obb_results.cls.cpu().numpy()):
                    x_center, y_center, width, height, rotation = obb[:5]
                    class_id = int(cls)
                    confidence = float(conf)

                    # Draw rotated bounding box using OpenCV
                    rect = ((x_center, y_center), (width, height), rotation * 180.0 / np.pi)
                    box_points = cv2.boxPoints(rect)
                    box_points = np.int0(box_points)
                    color = colors.get(class_id, (0, 255, 0))
                    cv2.drawContours(frame, [box_points], 0, color, 1)  # Set thickness to 1
                    print(f"Drawn OBB for class_id {class_id} with confidence {confidence} in frame {frame_count}.")

                    # Draw label with appropriate position and reduced thickness
                    label = class_names.get(class_id, 'unknown')
                    text_position = (int(x_center), int(y_center) - int(height / 2) - 10)
                    cv2.putText(frame, f"{label}: {confidence:.2f}",
                                text_position,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  # Set font thickness to 1

                    # Count detected objects
                    overall_counts[label] = overall_counts.get(label, 0) + 1

            frames.append(frame)
            print(f"Frame {frame_count} processed.")

        cap.release()
        print("Video processing completed.")

        # Save processed video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            output_path = tmpfile.name
        print(f"Saving processed video to {output_path}")

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480))
        for idx, frame in enumerate(frames):
            out.write(frame)
            if idx % 100 == 0:
                print(f"Writing frame {idx} to video.")

        out.release()
        print("Video saved.")

        # Create DataFrame to store results
        df = pd.DataFrame({
            'Label': list(overall_counts.keys()),
            'Object Count': list(overall_counts.values())
        })
        print("DataFrame created.")

        return output_path, df

    except Exception as e:
        print(f"Error in detect_and_draw_video: {e}")
        # In case of an error, return the original video and an empty DataFrame
        return video_path, pd.DataFrame({
            'Label': [],
            'Object Count': []
        })

# Gradio interface for images
image_interface = gr.Interface(
    fn=detect_and_draw_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[gr.Image(type="pil", label="Processed Image"), gr.Dataframe(label="Object Counts")],
    title="Object Detection in Aerial Images",
    description="Upload an aerial image to see detected objects and their counts.",
    examples=[
        '/content/EXAMPLES/IMAGES/Examples_images_areial_car.jpg',
        '/content/EXAMPLES/IMAGES/Examples_images_images.jpg',
        '/content/EXAMPLES/IMAGES/Examples_images_t.jpg'
    ]
)

# Gradio interface for videos
video_interface = gr.Interface(
    fn=detect_and_draw_video,
    inputs=gr.Video(label="Upload Video"),
    outputs=[gr.Video(label="Processed Video"), gr.Dataframe(label="Object Counts")],
    title="Object Detection in Videos",
    description="Upload a video to see detected objects and their counts.",
    examples=[
        '/content/EXAMPLES/VIDEO/airplane.mp4',
        '/content/EXAMPLES/VIDEO/city.mp4'
    ]
)

# Launch the app using a tabbed interface
app = gr.TabbedInterface([image_interface, video_interface], ["Image Detection", "Video Detection"])
app.launch()
