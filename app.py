import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import cv2

st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🎯",
    layout="wide"
)

st.title("AI Object Detection System")
st.write(
    "Upload an image to detect objects using YOLO."
)


@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)

confidence = st.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.35,
    step=0.05
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    if st.button("Detect Objects", type="primary"):

        with st.spinner("Detecting objects..."):

            model = load_model()

            results = model.predict(
                source=image,
                conf=confidence,
                verbose=False
            )

            result = results[0]

            annotated = result.plot()
            annotated = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            )

            st.subheader("Detection Result")

            st.image(
                annotated,
                use_container_width=True
            )

            detections = []

            for box in result.boxes:

                class_id = int(box.cls[0].item())
                score = float(box.conf[0].item())

                detections.append({
                    "Object": result.names[class_id],
                    "Confidence (%)": round(
                        score * 100, 2
                    )
                })

            st.subheader("Detection Summary")

            if detections:
                df = pd.DataFrame(detections)

                st.metric(
                    "Objects detected",
                    len(detections)
                )

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

                output = Image.fromarray(annotated)

                import io
                buffer = io.BytesIO()
                output.save(buffer, format="PNG")

                st.download_button(
                    "Download detected image",
                    data=buffer.getvalue(),
                    file_name="detected_objects.png",
                    mime="image/png"
                )

            else:
                st.warning(
                    "No objects detected. "
                    "Try another image or lower "
                    "the confidence threshold."
                )
