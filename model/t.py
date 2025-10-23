import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx

mlp = joblib.load("model\embedding_scaler_extend.pkl")
initial_type = [('float_input', FloatTensorType([None, len(mlp.coefs_[0])]))]
onnx_model = convert_sklearn(mlp, initial_types=initial_type)
onnx.save_model(onnx_model, "mlp_speed_classifier_extend.onnx")
