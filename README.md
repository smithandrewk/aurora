# Aurora Pipeline

Directory Structure
```
aurora
├── main.ipynb
├── makefile
├── README.md
├── requirements.txt
├── scripts
└── utils
```

## Setup
<br />
Before starting, rename data archive 

```Unscored.zip```

And rename ZDB archive

```UnscoredZDB```

<br />
NOTE: Zdb files must be scored at least onces for process to be successful</br>

<br />

## Scoring
Follow steps in
```main.ipynb```
to score data

<br />

## Output

Final output scorings are in

```data/final_ann```

```data/final_rf```

Final output scoring in zdb format are in

```data/ZDB_final_ann```

```data/ZDB_final_rf```

1. Raw Data exported from Neuroscore in xlsx format
2. Run pipeline, scoring raw data with Random Forest and Artificial Neural Network
3. Add scoring data to raw .zdb file using MATLAB script
## Installation
## Usage
