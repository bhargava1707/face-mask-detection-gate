import argparse
import os
import cv2
import numpy as np
from PIL import Image

from utils.model_io import load_tf_model

# Default model path (change if needed)
DEFAULT_MODEL_PATH = os.path.join("models", "face_mask_detection.pb")

# A super-light placeholder inference if TF/model isn't available
def fake_inference(image_bgr):
    """
    Draws a dummy box + label 'Mask (demo)' so you can see the pipeline working
    without a real model. Returns (boxes, labels).
    """
    h, w = image_bgr.shape[:2]
    # Make a centered rectangle for demo
    box_w, box_h = int(w * 0.3), int(h * 0.3)
    x1 = (w - box_w) // 2
    y1 = (h - box_h) // 2
    x2 = x1 + box_w
    y2 = y1 + box_h
    return [(x1, y1, x2, y2)], ["Mask (demo)"]

def run_inference(sess, graph, image_bgr):
    """
    If you wire your real graph here, map input/output tensors and run session.
    For now, we fall back to fake_inference if sess/graph is None.
    """
    if sess is None or graph is None:
        return fake_inference(image_bgr)

    # TODO: Replace with your real tensor names and preprocessing.
    # Example skeleton:
    # input_tensor = graph.get_tensor_by_name("input:0")
    # boxes_tensor = graph.get_tensor_by_name("boxes:0")
    # labels_tensor = graph.get_tensor_by_name("labels:0")
    # out_boxes, out_labels = sess.run([boxes_tensor, labels_tensor], feed_dict={input_tensor: batch})
    # return out_boxes, out_labels

    # Until you update the above, use demo inference:
    return fake_inference(image_bgr)

def draw_detections(image_bgr, boxes, labels):
    for (x1, y1, x2, y2), label in zip(boxes, labels):
        color = (0, 255, 0) if "Mask" in label else (0, 0, 255)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_bgr, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image_bgr

def main():
    parser = argparse.ArgumentParser(description="Face Mask Detection (Image Mode)")
    parser.add_argument("--img-path", type=str, required=True, help="Path to an image file")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to .pb model")
    args = parser.parse_args()

    # Try to load TF model (optional)
    sess, graph = load_tf_model(args.model_path)

    if not os.path.isfile(args.img_path):
        raise FileNotFoundError(f"Image not found: {args.img_path}")

    image_bgr = cv2.imread(args.img_path)
    if image_bgr is None:
        raise RuntimeError("Failed to read image. Check the file format or path.")

    boxes, labels = run_inference(sess, graph, image_bgr)
    out = draw_detections(image_bgr, boxes, labels)

    cv2.imshow("Mask Detection", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
