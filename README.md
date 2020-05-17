# debiasing-images
Project for CS 335.


# Install dependencies
```
pip install -r requirements.txt
```

# Run training
```
python src/train.py
```

Example with more command line parameters
```
python src/train.py --out-dir output --subset-percentage 0.01 --batch-size 16 -lr 0.0001 --hidden-size 512
```

# To train the baseline model
```
python src/train.py --out-dir output --subset-percentage 1.0 --batch-size 32 -lr 0.00001 --hidden-size 512 --num-epochs 10 --baseline
```


# Evaluate model
```
python src/test.py --weights output/best.pkl
```

# Pre-trained weights

Baseline model: [Google Drive](https://drive.google.com/file/d/1p-gH5-JYwBkVf7aObgdKWOwHgsBWxyo_/view?usp=sharing)
Our model: Coming soon
