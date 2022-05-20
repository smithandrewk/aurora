default: ## Print Makefile help
	@grep -Eh '^[a-z.A-Z_0-9-]+:.*?## .*$$' ${MAKEFILE_LIST} | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'
clean: ## Remove Data Things
	cowsay cleaning
	rm -rf data/raw data/renamed data/mapping

openZIP:	
	mkdir -p data/raw
	cp Unscored.zip data/Unscored.zip
	unzip -j data/Unscored.zip -d ./data/raw

downloadmodels:
	mkdir -p model
	xdg-open https://drive.google.com/drive/folders/1SwuAuVgyNirb-ebVKrU9RIzX6RFrhbdO?usp=sharing

renameZDB:
	./scripts/unzipAndRenameZDBData.py

scoreData:
	chmod +x main.py
	./main.py

archiveScores:
	cp -r data/final_ann .
	cp -r data/final_rf .
	zip -r Scored.zip final_ann/ final_rf/
	cp Scored.zip data/Scored.zip
	rm -r final_ann final_rf

createDB:
	python3 -c "from app import db; db.create_all()"