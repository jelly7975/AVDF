## Dependencies
The code is tested with Python=3.9.5 and pytorch=1.9.0. To install required packages:
```
pip install -r requirements.txt
```

## Training
For training, run:
```python
python main.py
```
You can optionally specify the following arguments:
* \-\-pretrain_path = AVDF.pth.par
* \-\-result_path = path where the results and trained models will be saved
* \-\-annotation_path = path to annotation file generated at previous step

## Testing
For testing, run:
```python
python main.py  --no_train --no_val
```