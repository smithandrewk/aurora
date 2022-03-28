default:
	cowsay empty
fd:
	cowsay making data
	./scripts/unzipAndRenameData.py
data:
	mkdir data
	mkdir figures
	xdg-open https://drive.google.com/file/d/1oMIHO4kANgirTFNrMLrxOKu1secCs9d_/view?usp=sharing
clean:
	rm figures/*
delstuff:
	rm -r data/preprocessed data/renamed data/balanced data/windowed data/mapping