import cv2
import numpy as np
import streamlit as st
from PIL import Image

from quality import check_image_quality
from detector import TowerDetector, calculate_average_confidence
from config import MODEL_PATH, CONFIDENCE_THRESHOLD


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Tower Detection",
    page_icon="📡",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📡 AI Tower Detection System")
st.write(
    "Upload an image to check its quality and detect "
    "monopole and supporting towers."
)


# --------------------------------------------------
# Upload image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload tower image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Read uploaded image
    pil_image = Image.open(uploaded_file).convert("RGB")

    # Convert RGB → BGR for OpenCV
    image = cv2.cvtColor(
        np.array(pil_image),
        cv2.COLOR_RGB2BGR
    )


    # --------------------------------------------------
    # Image Quality Check
    # --------------------------------------------------

    st.subheader("1. Image Quality Assessment")

    quality = check_image_quality(image)


    # Display quality metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Blur Score",
        f"{quality['blur_score']:.2f}"
    )

    col2.metric(
        "Brightness",
        f"{quality['mean_brightness']:.2f}"
    )

    col3.metric(
        "Underexposed",
        f"{quality['underexposed_ratio']:.2%}"
    )

    col4.metric(
        "Overexposed",
        f"{quality['overexposed_ratio']:.2%}"
    )


    # --------------------------------------------------
    # Reject bad image
    # --------------------------------------------------

    if not quality["acceptable"]:

        st.error("❌ Image rejected")

        st.write("Quality problems detected:")

        for reason in quality["reasons"]:
            st.write(f"- {reason}")

        st.warning(
            "YOLO detection was not executed because "
            "the image failed the quality check."
        )

        st.stop()


    # --------------------------------------------------
    # Accept image
    # --------------------------------------------------

    st.success("✅ Image quality acceptable")

    st.subheader("2. YOLO Tower Detection")


    # Load trained model
    detector = TowerDetector(MODEL_PATH)


    # Run detection
    result, detections = detector.predict(
        image,
        confidence=CONFIDENCE_THRESHOLD
    )


    # --------------------------------------------------
    # Display detection image
    # --------------------------------------------------

    annotated_image = result.plot()

    annotated_image = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_BGR2RGB
    )

    st.image(
        annotated_image,
        caption="Detected towers",
        use_container_width=True
    )


    # --------------------------------------------------
    # Detection results
    # --------------------------------------------------

    if not detections:

        st.warning("No towers detected.")

    else:

        st.subheader("3. Detection Results")


        # Detection table
        rows = []

        for i, detection in enumerate(
            detections,
            start=1
        ):

            rows.append({
                "Detection": i,
                "Class": detection["class_name"],
                "Confidence": f"{detection['confidence']:.2%}",
                "X1": round(detection["bbox"][0], 1),
                "Y1": round(detection["bbox"][1], 1),
                "X2": round(detection["bbox"][2], 1),
                "Y2": round(detection["bbox"][3], 1),
            })


        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True
        )


        # --------------------------------------------------
        # Average confidence
        # --------------------------------------------------

        st.subheader("4. Average Confidence by Class")

        average_confidence = calculate_average_confidence(
            detections
        )


        cols = st.columns(
            max(1, len(average_confidence))
        )


        for col, (class_name, confidence) in zip(
            cols,
            average_confidence.items()
        ):

            col.metric(
                class_name,
                f"{confidence:.2%}"
            )