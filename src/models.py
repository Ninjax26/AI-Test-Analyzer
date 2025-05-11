from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.String(128), primary_key=True)
    name = db.Column(db.String(256))
    email = db.Column(db.String(256), unique=True)
    avatar = db.Column(db.String(512)) 