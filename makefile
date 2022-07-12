createDB:
	python3 -c "from app import db; db.create_all()"
docker:
	sudo docker build -t aurora:server .
	sudo docker run -dtp 80:5000 aurora:server
