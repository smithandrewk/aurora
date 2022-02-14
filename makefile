default: ## Print Makefile help
	@grep -Eh '^[a-z.A-Z_0-9-]+:.*?## .*$$' ${MAKEFILE_LIST} | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'
downloaddata: ## Download Sleep Data from Drive
	xdg-open https://drive.google.com/file/d/1BLG-DsxcSmPPex32xt7m_ayLb2T30-X3/view?usp=sharing
rename: ## Unzip Data and Rename
	cowsay making data
	./scripts/unzipAndRenameData.py
renamenewdata: ## Unzip and Rename New Data
	cowsay making data
	./scripts/unzipAndRenameNewData.py
clean: ## Remove Data Things
	cowsay cleaning
	rm -rf data/raw data/renamed data/mapping


renameZIP:	
	./scripts/unzipAndRenameZipFile.py

renameNoZIP:
	./scripts/renameData.py

#downloadmodels:
#	mkdir -p ./model
#	wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1MSAECbvmlsptmRDdR1EEqo3ijRZrf9gD' -O model/best_model.h5
#	wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1GW09FZXP10H_2pJnVZH_UKsmLx-PMtpV' -O model/best_transformer.h5
#	wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Sx7ZVgD7gyqfjZJnweEIdzr8h4-7HZfw' -O model/saved_model.pb
#	wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Sx7ZVgD7gyqfjZJnweEIdzr8h4-7HZfw' -O model/saved_model.pb

downloadmodels:
	mkdir -p model
	xdg-open https://drive.google.com/drive/folders/1SwuAuVgyNirb-ebVKrU9RIzX6RFrhbdO?usp=sharing

renameZDB:
	./scripts/unzipAndRenameZDBData.py
