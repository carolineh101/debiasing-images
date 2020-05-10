# debiasing-images
Project for CS 335.


# Install dependencies
```
pip install -r requirements.txt
```

# Run training
```
python src/train.py --out-dir output
```

Example with more command line parameters
```
!python src/train.py --out-dir output --subset-percentage 0.01 --batch-size 64 -lr 0.01 --hidden_size 512
```


# Evaluate model
```
python src/test.py --weights output/best.pkl
```

# Pre-trained weights

Baseline model: [Google Drive](https://drive.google.com/file/d/1p-gH5-JYwBkVf7aObgdKWOwHgsBWxyo_/view?usp=sharing)
Our model: Coming soon
