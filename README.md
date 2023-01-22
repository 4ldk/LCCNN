# LCCNN

CNN models for [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/ ) and [Open-Source-Event-Driven-ECG-Dataset](https://github.com/jedaiproject/Open-Source-Event-Driven-ECG-Dataset) by PyTorch.


## How to train

1. Download [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/ ) and [Open-Source-Event-Driven-ECG-Dataset](https://github.com/jedaiproject/Open-Source-Event-Driven-ECG-Dataset).
2. `pip install -r requirements.txt`
3. Rewrite `setting_path.py`.
4. Run `preprocessing.py`
5. Rewrite and Run 'train.py'


## Models
- CNN
  - Simple CNN model for MIT-BIH Arrhythmia Database designed in [4].
- LCCNN
  - Simple CNN model for Open-Source-Event-Driven-ECG-Dataset designed in [4].
- LCCNNLight
  - CNN model for Open-Source-Event-Driven-ECG-Dataset lighter than LCCNN by using additional CNN layer.
- Resnet34
  - 1d-ResNet34 for MIT-BITH Arrhythmia Database
  - You can use this for Open-Source-Event-Driven-ECG-Dataset by small change.
  
## Input Layers
- NormalCNN
  - Simple 2ch-CNN layer for Open-Source-Event-Driven-ECG-Dataset designed in [4].
- TimeEmbedding
  - Input layer for Open-Source-Event-Driven-ECG-Dataset that is 1ch-CNN layer + full-learnable positional encoding layer.
- TimeSin
  - Input layer for Open-Source-Event-Driven-ECG-Dataset that is 1ch-CNN layer + sinusoidal positional encoding layer.
- NonTimed
  - 1ch-CNN layer for Open-Source-Event-Driven-ECG-Dataset.

## References

[1] https://physionet.org/content/mitdb/1.0.0/  
[2] Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)  
[3] Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.  
[4] M. Saeed et al., "Evaluation of Level-Crossing ADCs for Event-Driven ECG Classification," in IEEE Transactions on Biomedical Circuits and Systems, vol. 15, no. 6, pp. 1129-1139, Dec. 2021, doi: 10.1109/TBCAS.2021.3136206.
