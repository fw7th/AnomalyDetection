from onnxruntime.quantization import quantize_dynamic, QuantType
import os


OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
    "models"
)

# Change the working directory temporarily to save in desired location
os.chdir(OUTPUT_DIR)

model_fp32 = "yolov8n.onnx"
model_int8 = "yolov8n_int8.onnx"

quantize_dynamic(
    model_fp32,
    model_int8,
    weight_type=QuantType.QInt8  # or QuantType.QUInt8
)
