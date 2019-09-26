import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

export_dir = './saved_model/2'
graph_pb = 'model/frozen_inference_graph_face.pb'

builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

with tf.gfile.GFile(graph_pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

sigs = {}

with tf.Session(graph=tf.Graph()) as sess:
    # name="" is important to ensure we don't get spurious prefixing
    tf.import_graph_def(graph_def, name="")
    g = tf.get_default_graph()
    inp = g.get_tensor_by_name("image_tensor:0")
    boxes = g.get_tensor_by_name("detection_boxes:0")
    num = g.get_tensor_by_name("num_detections:0")
    scores = g.get_tensor_by_name("detection_scores:0")
    classes = g.get_tensor_by_name("detection_classes:0")


    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.saved_model.signature_def_utils.predict_signature_def(
            {"image": inp}, 
            {
                "boxes": boxes, 
                "num_detections": num, 
                "scores": scores,
                "classes": classes
            })

    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map=sigs)

builder.save()