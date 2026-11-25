"""
Módulos de inferencia de la Estación 2 (Edge / Raspberry Pi y PC).

- preprocessing.py  Preprocesamiento OpenCV/NumPy y softmax.
- predictor.py      QualityPredictor (ONNX Runtime / PyTorch opcional).
- oranges.py        Nombres de fotos, agrupación y promedio por naranja.
- camera_usb.py     FastCameraReader (USB / V4L2).

La CLI sigue en src/inference.py, que además reexporta estos nombres.
"""
