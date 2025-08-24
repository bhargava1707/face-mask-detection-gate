import argparse
import os
import cv2
from utils.model_io import load_tf_model
from mask_detection import run_inference, draw_detections, DEFAULT_MODEL_PATH

def main():
    parser = argparse.ArgumentParser(description="Face Mask Detection (Video/Camera)")
    parser.add_argument("--video-path", type=str, default="0", help="Path to video file, or '0' for webcam")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to .pb model")
    args = parser.parse_args()

    sess, graph = load_tf_model(args.model_path)

    # Camera index if "0"
    cap_src = 0 if args.video_path == "0" else args.video_path
    cap = cv2.VideoCapture(cap_src)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.video_path}")

    print("[run_video] Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        boxes, labels = run_inference(sess, graph, frame)
        out = draw_detections(frame, boxes, labels)

        cv2.imshow("Mask Detection (Video)", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()