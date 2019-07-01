from keras import backend as K
from keras.models import load_model
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants 
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
def export_h5_to_pb(path_to_h5="best_model.hdf5", export_path="best_model/1"):
    # Set the learning phase to Test since the model is already trained.
    K.set_learning_phase(0)
    print(path_to_h5)
    # Load the Keras model
    keras_model = load_model(path_to_h5)
    print(path_to_h5)
    # Build the Protocol Buffer SavedModel at 'export_path'
    builder = saved_model_builder.SavedModelBuilder(export_path)

    # Create prediction signature to be used by TensorFlow Serving Predict API
    signature = predict_signature_def(inputs={"input_traces": keras_model.input},
                                      outputs={"output_trace": keras_model.output})
                                      
    print(keras_model.input,keras_model.output)

    with K.get_session() as sess:
        # Save the meta graph and the variables
        builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING],
                                         signature_def_map={"predict": signature})

    builder.save()


if __name__ == '__main__':
    export_h5_to_pb()
