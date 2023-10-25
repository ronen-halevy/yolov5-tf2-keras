convert_weights_pt2k.py

Coverts weights from pytorch model to keras model.

Input: pytorch .pt model file

Outputs:
1. keras model .pb files, with weights.
2. keras model .tf files weights only.

Description
+++++++++++++
converts .pt pytorch model file to keras model .pb files and keras weights .tf files.

command line:
+++++++++++++

python convert_weights_t2k  --weights wpath --imgsz size  --tf_weights_dir wdir --tf_model_dir mdir

defaults:
+++++++++
imgsz=640

