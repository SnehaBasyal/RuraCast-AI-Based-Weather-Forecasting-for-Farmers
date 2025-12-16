# models.py
from extensions import db
from flask_login import UserMixin
from datetime import datetime

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    country_code = db.Column(db.String(5), nullable=False)
    mobile = db.Column(db.String(15), nullable=False)
    password = db.Column(db.String(200), nullable=False)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    address = db.Column(db.String(200), nullable=False)
    forecast_days = db.Column(db.Integer, nullable=False)
    language = db.Column(db.String(20), nullable=False)
    pdf_path = db.Column(db.String(200), nullable=False)
    png_path = db.Column(db.String(200), nullable=False)
    most_recommended_crop = db.Column(db.String(50), nullable=False)
    translated_crop_name = db.Column(db.String(50), nullable=False)
    rain_summary = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)