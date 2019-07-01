
from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import time

from grpc.beta import implementations

from protos import tensor_pb2
from protos import tensor_shape_pb2
from protos import types_pb2
from protos import prediction_service_pb2
from protos import predict_pb2

#from utils.processdata import processdata
#from utils.utils import load_data_w_sim_noise, shuffle_data

root_dir = "/home/dominik/workdir/datadir/"

tt = time.time()

parser = argparse.ArgumentParser(description='incetion grpc client flags.')
parser.add_argument('--host', default='0.0.0.0', help='inception serving host')
parser.add_argument('--port', default='8500', help='inception serving port')
# parser.add_argument('--image', default='', help='path to JPEG image file')
FLAGS = parser.parse_args()


def plot_traces(x_noisy, x_label, x_pred):
  n = 4  # how many digits we will display
  # plt.figure(figsize=(30, 4))
  for i in range(n):
    # display original
    ax = plt.subplot(3, 4, i + 1)

    plt.plot(x_noisy[i])
    plt.gray()
    # plt.title("noisy_traces")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, 4, i + 1 + n)
    plt.plot(x_label[i])
    plt.gray()
    # plt.title("true_signal")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, 4, i + 1 + 2 * n)
    plt.plot(x_pred[i])
    plt.gray()
    # plt.title("predicted_traces")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

  plt.savefig("/home/dominik/workdir/datadir/client.png")
  plt.close()



def main():
    # create prediction service client stub
    channel = implementations.insecure_channel(FLAGS.host, int(FLAGS.port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # create request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'resnet'
    request.model_spec.signature_name = 'serving_default'

    # read image into numpy array
    data = [1]*1000

    # convert to tensor proto and make request
    # shape is in NHWC (num_samples x height x width x channels) format
    tensor_shape = [1, 1000, 1, 1]
    dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=dim) for dim in tensor_shape]
    tensor_shape = tensor_shape_pb2.TensorShapeProto(dim=dims)
    tensor = tensor_pb2.TensorProto(
        dtype=types_pb2.DT_FLOAT,
        tensor_shape=tensor_shape,
        float_val=list(data.reshape(-1)))


    request.inputs['inputs'].CopyFrom(tensor)

    resp = stub.Predict(request, 30.0)

    print('total time: {}s'.format(time.time() - tt))
    print(resp)


if __name__ == '__main__':
    main()


