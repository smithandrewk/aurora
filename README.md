# Aurora Pipeline to be used in Application
Directory Structure
```
aurora
├── main.py
├── .gitignore
├── README.md
├── requirements.txt
└── lib
```

## Setup
<br />
Before starting, move data files to `data/0_raw'
And move zdb files to `data/6_raw_zdb`


## Scoring
To score data, run the script

```main.py --ann-model [model in model directory]```

run `main.py -h` for help

<br />

## Output

Final output scorings are in

```data/5_final_lstn```

Final output scoring in zdb format are in

```data/9_final_zdb_lstm```

=======
1. Raw Data exported from Neuroscore in xlsx format
2. Run pipeline, scoring raw data with LSTM Artifical Neural Network
3. Open scored ZDB files with Neuroscore
## Installation
## Usage
## Contributing
[Making Changes to Project Aurora Pipeline on Git](https://andrewsmithnotion.notion.site/Making-Changes-to-Project-Aurora-Pipeline-on-Git-7fc6fbc74c33468cad0ed004fd6e4b5e)
