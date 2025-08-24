
import os

def load_tf_model(model_path: str):
    """
    Tries to load a frozen TensorFlow graph (.pb).
    Returns (sess, graph) if successful, else (None, None).
    """
    try:
        import tensorflow as tf  # type: ignore
    except Exception as e:
        print("[model_io] TensorFlow not available:", e)
        return None, None

    if not os.path.isfile(model_path):
        print(f"[model_io] Model file not found at: {model_path}")
        return None, None

    try:
        graph = tf.Graph()
        with graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, "rb") as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")
        sess = tf.compat.v1.Session(graph=graph)
        print("[model_io] Loaded TensorFlow graph successfully.")
        return sess, graph
    except Exception as e:
        print("[model_io] Failed to load model:", e)
        return None, None
