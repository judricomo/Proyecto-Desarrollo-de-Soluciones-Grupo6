import streamlit as st
import torch
from PIL import Image, ImageDraw
import pandas as pd
import time

# Add inference logic
def detection(image):

    return {
        "images": [
            "S_11_05_16_DSC01556.JPG",
            "S_11_05_16_DSC01556.JPG",
            "S_11_05_16_DSC01556.JPG",
            "S_11_05_16_DSC01556.JPG",
            "S_11_05_16_DSC01556.JPG",
            "S_11_05_16_DSC01556.JPG",
            "S_11_05_16_DSC01556.JPG",
            "S_11_05_16_DSC01556.JPG",
            "S_11_05_16_DSC01556.JPG",
            "S_11_05_16_DSC01556.JPG",
            "S_11_05_16_DSC01556.JPG",
            "S_11_05_16_DSC01556.JPG",
            "S_11_05_16_DSC01556.JPG",
            "S_11_05_16_DSC01556.JPG",
            "S_11_05_16_DSC01556.JPG",
            "S_11_05_16_DSC01556.JPG",
            "S_11_05_16_DSC01556.JPG",
            "S_11_05_16_DSC01556.JPG",
            "S_11_05_16_DSC01556.JPG",
        ],
        "x_min": [
            1820,
            1768,
            1649,
            1600,
            1594,
            1603,
            1330,
            1325,
            1255,
            1322,
            1242,
            1188,
            1184,
            1282,
            1190,
            1205,
            947,
            767,
            799,
        ],
        "y_min": [
            1017,
            1026,
            1035,
            1001,
            1033,
            1060,
            1009,
            1046,
            990,
            733,
            824,
            887,
            820,
            571,
            590,
            659,
            803,
            936,
            1070,
        ],
        "x_max": [
            1883,
            1820,
            1734,
            1666,
            1655,
            1655,
            1408,
            1378,
            1345,
            1377,
            1337,
            1250,
            1248,
            1361,
            1248,
            1259,
            1011,
            841,
            894,
        ],
        "y_max": [
            1081,
            1055,
            1094,
            1049,
            1067,
            1122,
            1048,
            1073,
            1046,
            811,
            873,
            936,
            870,
            605,
            624,
            718,
            850,
            1013,
            1106,
        ],
        "labels": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    }


def page_single_image():
    st.subheader("Single Image Detection 🦊")
    st.write("1. Upload your aerial image 📤")

    uploaded_file = st.file_uploader(
        "Upload an image:", type=["jpg", "jpeg", "png"], key="single"
    )

    if uploaded_file is not None:
        st.toast("Image uploaded! 📷")
        image_original = Image.open(uploaded_file)

        col1, col2 = st.columns(2)
        col1.image(image_original, caption="Original Image", use_container_width=True)

        st.write("2. Recognize animals 🦊")
        if st.button("Recognize", key="single_recognize"):
            st.toast("Detecting animals... 🕵️")
            time.sleep(6)
            image_detected = image_original.copy()
            draw = ImageDraw.Draw(image_detected)

            # Run detection
            detections = detection(image_original)
            df = pd.DataFrame(detections)
            df['x'] = ((df['x_min'] + df['x_max']) / 2).astype(float)
            df['y'] = ((df['y_min'] + df['y_max']) / 2).astype(float)
            df_1 = df[['images', 'x', 'y', 'labels']]

            # Draw circles for each detection
            for idx, row in df_1.iterrows():
                center_x = row['x']
                center_y = row['y']
                r = 15
                draw.ellipse(
                    (center_x - r, center_y - r, center_x + r, center_y + r),
                    fill = "red", outline = "red"
                )
            st.toast("Drawing image... 🎨")
            time.sleep(6)
            
            col2.image(
                image_detected, caption="Detected Animals", use_container_width=True
            )

            # Extra toasts for table creation
            st.toast("Creating detection details table... 🗂️")
            time.sleep(3)
            st.subheader("Detection Details")
            st.write("Below are the bounding box coordinates for each detected animal.")

            st.table(df_1)
            st.toast("Table created successfully! 🎉")


# def page_batch_prediction():
#     st.subheader("Batch Prediction 🗃️")
#     st.write(
#         "Upload multiple images or specify a local folder to detect animals in each one."
#     )

#     # 1) File uploader
#     batch_files = st.file_uploader(
#         "Select multiple images:",
#         type=["jpg", "jpeg", "png"],
#         accept_multiple_files=True,
#         key="batch",
#     )

#     # 2) Local folder input
#     folder_path = st.text_input("Or enter a local folder path containing images:")
#     if st.button("Load from local folder"):
#         import os

#         if os.path.isdir(folder_path):
#             local_images = []
#             for file_name in os.listdir(folder_path):
#                 ext = file_name.lower().split(".")[-1]
#                 if ext in ["jpg", "jpeg", "png"]:
#                     full_path = os.path.join(folder_path, file_name)
#                     local_images.append(full_path)
#             # Store in session_state for usage later
#             st.session_state["local_images"] = local_images
#             st.toast(f"Loaded {len(local_images)} images from folder! 📂")
#         else:
#             st.toast("Invalid folder path!")

#     # Combine images from file_uploader and local_images if any
#     all_image_sources = list(batch_files) if batch_files else []
#     if "local_images" in st.session_state:
#         all_image_sources.extend(st.session_state["local_images"])

#     if all_image_sources:
#         st.toast("Images ready for batch detection!")
#         if st.button("Recognize All", key="batch_recognize"):
#             st.toast("Detecting animals in batch... 🕵️")

#             all_detections = []

#             import os

#             for src in all_image_sources:
#                 if isinstance(src, str):
#                     # This is a file path
#                     image_path = src
#                     image_name = os.path.basename(src)
#                     image = Image.open(src)
#                 else:
#                     # This is an UploadedFile
#                     image_name = src.name
#                     image = Image.open(src)

#                 detections = simulate_detection(image)
#                 # For each detection, store a row in the table data
#                 for det in detections:
#                     x1, y1, x2, y2 = det["bbox"]
#                     row = {
#                         "image name": image_name,
#                         "x1": x1,
#                         "y1": y1,
#                         "x2": x2,
#                         "y2": y2,
#                         "label": det["label"],
#                     }
#                     all_detections.append(row)

#             df = pd.DataFrame(all_detections)

#             st.toast("Batch detection completed! 🎉")

#             # Display table
#             st.subheader("Batch Detection Results")
#             st.write("Below is a table of all detections across your uploaded images.")
#             st.table(df)

#             # Download as CSV
#             csv_data = df.to_csv(index=False)
#             st.download_button(
#                 label="Download Results as CSV",
#                 data=csv_data,
#                 file_name="batch_detections.csv",
#                 mime="text/csv",
#             )


def main():
    st.set_page_config(page_title="Aerial Animal Detector", page_icon="🦁")

    st.title("Aerial Animal Detector 🦁")
    st.write("Easily detect animals in your aerial images. Click a page on the left.")

    # Create a sidebar menu with direct links
    st.sidebar.title("Navigation 🧭")

    if st.sidebar.button("Single Image 🦊"):
        st.session_state.page = "single"
    # if st.sidebar.button("Batch Prediction 🗃️"):
    #     st.session_state.page = "batch"

    # Default if no session state is set
    if "page" not in st.session_state:
        st.session_state.page = "single"

    if st.session_state.page == "single":
        page_single_image()
    # else:
    #     page_batch_prediction()


if __name__ == "__main__":
    main()
