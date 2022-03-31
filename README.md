# Training
A branch of project aurora used for training neural networks
## Usage
### Clone Repository
```
git clone https://github.com/smithandrewk/aurora.git
cd aurora
```
### Checkout Training Branch
```
git checkout training
```
### Download Training Data
1. Make data directory in aurora
```
mkdir data
```
2. Make raw directory in aurora/data

```
mkdir data/raw
```
3. Download training data
4. Move *.xls training files into **aurora/data/raw**
5. **Verify there are only \*.xls training files in aurora/data/raw**
6. Run main.py

```
python3 main.py
```

### Options for main.py:
Notes: 
- Can run new-dir, split-and-shuffle, and train-model together to do all
- If `--split-and-shuffle` and `--train-model` are specified with `--data-dir`, model will be trained on new data created after split and shuffle

See help
```
./main.py --help
```
Create a new data set from data in `data/raw`
```
./main.py --new-dir
```
Split and shuffle data
```
./main.py --split-and-shuffle
```
Train model
```
./main.py --train-model
```
Use data set from previous session for split and shuffle or for training model
```
./main.py --data-dir [MM.DD.YYYY_hh:mm]
```
skip certain features
```
./main.py --skip-features [Feature] [Feature] ...
```
select certain features
```
./main.py --select-features [Feature] [Feature] ...
```