# Fine Tuning Mistral

## Overview

For fine-tuning mistral, we need dataset in Excel file. Then preprocessing would be done on the data, followed by its structuring. The data would then be provided to model which would be used for fine-tuning the model. When fine tuning is done, there will be new weights and that weights will be merge with original weights. 

## Instructions to follow

To begin with, these steps has to be performed: 

1. Create conda environment:
```
conda create -n mistral python=3.8
```
2. To activate this environment, run:
```
conda activate mistral
```
3. Install requirements:

```
pip install -r requirements.txt
```
4. Install pytorch: 
```
pip3 install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

5. Install flash attention
```
pip install flash-attn --no-build-isolation
```
6. To run the code , you need to run `main.py` file. 

```
python main.py --upload_file fitness_dataset.xlsx --nrows 1
```

- `--upload_file` : dataset path (format should xslx)
-  `--nrows` : number of rows used for training (Optional) 
7. Merging new weights with base model weights
```
python merge_model.py
```
8. Run inference using
```
python inference.py
```

## Working

The code would take data, preprocess it and feed to mistral model. The training would start and it would take some time. After training is finished, the new weights will be merged with base model weights and we will get new fine-tuned model. The fine tuned model can be used on new unseen data for generating responses 

## Error Handling

If you see an error like this

```
ImportError: Using `load_in_8bit=True` requires Accelerate: `pip install accelerate` and the latest version of bitsandbytes `pip install -i https://test.pypi.org/simple/ bitsandbytes` or pip install bitsandbytes`
```

Then you need to uninstall pytorch using 
```
pip uninstall torch
```
And install pytorch again using
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```