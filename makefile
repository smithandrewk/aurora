createDB:
	python3 -c "from app import db; db.create_all()"
docker:
	docker build -t aurora:server .
	docker run -dtp 80:5000 aurora:server