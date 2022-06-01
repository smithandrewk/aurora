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

```UnscoredZDB.zip```

<br />
NOTE:  zdb code has not been completely tested</br>
<br />

Prepare data for scoring by typing the following in the command line

```make openZIP```

and, if ZBD files are available,

```make renameZDB```

<br />

## Scoring
To score data, type the command

```make scoreData```

or run the script

```main.py```

<br />

## Output

Final output scorings are in

```data/final_ann```

```data/final_rf```

Final output scoring in zdb format are in

```data/ZDB_final_ann```

```data/ZDB_final_rf```

=======
1. Raw Data exported from Neuroscore in xlsx format
2. Run pipeline, scoring raw data with Random Forest and Artificial Neural Network
3. Add scoring data to raw .zdb file using MATLAB script
## Installation
## Usage
## Contributing
[Making Changes to Project Aurora Pipeline on Git](https://andrewsmithnotion.notion.site/Making-Changes-to-Project-Aurora-Pipeline-on-Git-7fc6fbc74c33468cad0ed004fd6e4b5e)
