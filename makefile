default:
	cowsay empty
	make fd
fd:
	cowsay making data
	./scripts/format_data.py
