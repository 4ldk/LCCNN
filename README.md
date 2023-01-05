# LCCNN

CNN models for [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/ ) and [Open-Source-Event-Driven-ECG-Dataset](https://github.com/jedaiproject/Open-Source-Event-Driven-ECG-Dataset) by PyTorch.


## How to train

1. Download [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/ ) and [Open-Source-Event-Driven-ECG-Dataset](https://github.com/jedaiproject/Open-Source-Event-Driven-ECG-Dataset).
2. `pip install -r requirements.txt`
3. Rewrite `setting_path.py`.
4. Run `preprocessing.py`
5. Rewrite and Run 'train.py'


## Models
- LSTM
  - Simple LSTM model for MIT-BIH Arrhythmia Database.
  - This is not reccomended.
- CNN
  - Simple CNN model for MIT-BIH Arrhythmia Database designed in [4].
- LCCNN
  - Simple CNN model for Open-Source-Event-Driven-ECG-Dataset designed in [4].
- TimeEmbeddingCNN
  - CNN model for Open-Source-Event-Driven-ECG-Dataset, but 1st layer is linear layer + full-learnable positional encoding layer.
- TimeSinCNN
  - CNN model for Open-Source-Event-Driven-ECG-Dataset, but 1st layer is linear layer + sinusoidal positional encoding layer.
- LCCNNLight
  - CNN model for Open-Source-Event-Driven-ECG-Dataset lighter than LCCNN by using big maxpooling.
- LCCNNLight2
  - CNN model for Open-Source-Event-Driven-ECG-Dataset lighter than LCCNN by using additional CNN layer.
- Resnet34
  - 1d-ResNet34 for MIT-BITH Arrhythmia Database
  - You can use this for Open-Source-Event-Driven-ECG-Dataset by small change.
  
## References

[1] https://physionet.org/content/mitdb/1.0.0/  
[2] Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)  
[3] Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.  
[4] M. Saeed et al., "Evaluation of Level-Crossing ADCs for Event-Driven ECG Classification," in IEEE Transactions on Biomedical Circuits and Systems, vol. 15, no. 6, pp. 1129-1139, Dec. 2021, doi: 10.1109/TBCAS.2021.3136206.
