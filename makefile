createDB:
	python3 -c "from app import db; db.create_all()"