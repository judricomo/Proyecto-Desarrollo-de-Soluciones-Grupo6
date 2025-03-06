import streamlit as st
import torch
from PIL import Image, ImageDraw
import pandas as pd

# Add inference logic
def simulate_detection(image):
    # Returns animals for demonstration.
    return [
        {'bbox': (50, 50, 100, 100), 'label': 'Elephant 🐘'},
        {'bbox': (150, 150, 200, 200), 'label': 'Lion 🦁'},
        {'bbox': (120, 60, 170, 110), 'label': 'Giraffe 🦒'},
        {'bbox': (60, 200, 100, 250), 'label': 'Lion 🦁'}, # extra sample
    ]

def page_single_image():
    st.subheader("Single Image Detection 🦊")
    st.write("1. Upload your aerial image 📤")

    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"], key="single")

    if uploaded_file is not None:
        st.toast("Image uploaded! 📷")
        image_original = Image.open(uploaded_file)

        col1, col2 = st.columns(2)
        col1.image(image_original, caption="Original Image", use_container_width=True)

        st.write("2. Recognize animals 🦊")
        if st.button("Recognize", key="single_recognize"):
            st.toast("Detecting animals... 🕵️")
            image_detected = image_original.copy()
            draw = ImageDraw.Draw(image_detected)

            # Run detection
            detections = simulate_detection(image_original)

            # Draw circles for each detection
            for detection in detections:
                bbox = detection['bbox']
                label = detection['label']

                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2

                dot_radius = 5
                x0 = center_x - dot_radius
                y0 = center_y - dot_radius
                x1 = center_x + dot_radius
                y1 = center_y + dot_radius

                draw.ellipse([x0, y0, x1, y1], fill="red")

                draw.text((center_x + dot_radius + 2, center_y - dot_radius), label, fill="red")

            col2.image(image_detected, caption="Detected Animals", use_container_width=True)
            num_detected = len(detections)
            st.toast(f"Detected {num_detected} animals! 🦁")

            # Extra toasts for table creation
            st.toast("Creating detection details table... 🗂️")

            st.subheader("Detection Details")
            st.write("Below are the bounding box coordinates for each detected animal.")

            detection_table = []
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                detection_table.append({
                    "Class": det["label"],
                    "Bounding Box (x1,y1,x2,y2)": f"({x1},{y1},{x2},{y2})"
                })

            st.table(detection_table)
            st.toast("Table created successfully! 🎉")

def page_batch_prediction():
    st.subheader("Batch Prediction 🗃️")
    st.write("Upload multiple images or specify a local folder to detect animals in each one.")

    # 1) File uploader
    batch_files = st.file_uploader(
        "Select multiple images:",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch"
    )

    # 2) Local folder input
    folder_path = st.text_input("Or enter a local folder path containing images:")
    if st.button("Load from local folder"):
        import os
        if os.path.isdir(folder_path):
            local_images = []
            for file_name in os.listdir(folder_path):
                ext = file_name.lower().split('.')[-1]
                if ext in ["jpg", "jpeg", "png"]:
                    full_path = os.path.join(folder_path, file_name)
                    local_images.append(full_path)
            # Store in session_state for usage later
            st.session_state['local_images'] = local_images
            st.toast(f"Loaded {len(local_images)} images from folder! 📂")
        else:
            st.toast("Invalid folder path!")

    # Combine images from file_uploader and local_images if any
    all_image_sources = list(batch_files) if batch_files else []
    if "local_images" in st.session_state:
        all_image_sources.extend(st.session_state['local_images'])

    if all_image_sources:
        st.toast("Images ready for batch detection!")
        if st.button("Recognize All", key="batch_recognize"):
            st.toast("Detecting animals in batch... 🕵️")

            all_detections = []

            import os
            for src in all_image_sources:
                if isinstance(src, str):
                    # This is a file path
                    image_path = src
                    image_name = os.path.basename(src)
                    image = Image.open(src)
                else:
                    # This is an UploadedFile
                    image_name = src.name
                    image = Image.open(src)

                detections = simulate_detection(image)
                # For each detection, store a row in the table data
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    row = {
                        "image name": image_name,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "label": det['label']
                    }
                    all_detections.append(row)

            df = pd.DataFrame(all_detections)

            st.toast("Batch detection completed! 🎉")

            # Display table
            st.subheader("Batch Detection Results")
            st.write("Below is a table of all detections across your uploaded images.")
            st.table(df)

            # Download as CSV
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv_data,
                file_name="batch_detections.csv",
                mime="text/csv"
            )

def main():
    st.set_page_config(page_title="Aerial Animal Detector", page_icon="🦁")

    st.title("Aerial Animal Detector 🦁")
    st.write("Easily detect animals in your aerial images. Click a page on the left.")

    # Create a sidebar menu with direct links
    st.sidebar.title("Navigation 🧭")

    if st.sidebar.button("Single Image 🦊"):
        st.session_state.page = "single"
    if st.sidebar.button("Batch Prediction 🗃️"):
        st.session_state.page = "batch"

    # Default if no session state is set
    if "page" not in st.session_state:
        st.session_state.page = "single"

    if st.session_state.page == "single":
        page_single_image()
    else:
        page_batch_prediction()

if __name__ == "__main__":
    main()
