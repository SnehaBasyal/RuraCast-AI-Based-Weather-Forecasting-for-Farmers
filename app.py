# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import json
import uuid
from extensions import db, login_manager, mail
from models import User, Prediction
from weather_utils import get_weather_forecast
from crop_data import get_crop_recommendation, CROP_NAMES, get_translated_crop_name
from notifications import send_email, send_sms

app = Flask(__name__)
app.config.from_pyfile('config.py')

# Initialize extensions with app
db.init_app(app)
login_manager.init_app(app)
mail.init_app(app)
login_manager.login_view = 'login'

# Import models after initializing db
from models import User, Prediction

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        email = request.form['email']
        country_code = request.form['country_code']
        mobile = request.form['mobile']
        password = request.form['password']
        
        # Check if user exists
        if db.session.execute(db.select(User).where(User.username == username)).scalar_one_or_none():
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
            
        if db.session.execute(db.select(User).where(User.email == email)).scalar_one_or_none():
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
            
        # Create new user
        new_user = User(
            name=name,
            username=username,
            email=email,
            country_code=country_code,
            mobile=mobile,
            password=generate_password_hash(password, method='pbkdf2:sha256')
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = db.session.execute(db.select(User).where(User.username == username)).scalar_one_or_none()
        
        if not user or not check_password_hash(user.password, password):
            flash('Invalid username or password', 'danger')
            return redirect(url_for('login'))
            
        login_user(user)
        return redirect(url_for('dashboard'))
        
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', name=current_user.name)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        name = request.form['name']
        address = request.form['address']
        forecast_days = int(request.form['forecast_days'])
        language = request.form['language']
        
        # Get weather forecast
        try:
            forecast_data, pdf_path, png_path = get_weather_forecast(address, forecast_days)
        except Exception as e:
            flash(f'Error getting weather forecast: {str(e)}', 'danger')
            return redirect(url_for('predict'))
        
        # Get crop recommendations
        crop_recommendations = []
        for day in forecast_data:
            crops = get_crop_recommendation(day)
            crop_recommendations.append(crops)
        
        # Find most recommended crop
        all_crops = [crop for day in crop_recommendations for crop in day]
        most_recommended = max(set(all_crops), key=all_crops.count)
        
        # Get translated crop name
        translated_crop_name = get_translated_crop_name(most_recommended, language)
        
        # Create prediction record
        prediction = Prediction(
            user_id=current_user.id,
            name=name,
            address=address,
            forecast_days=forecast_days,
            language=language,
            pdf_path=pdf_path,
            png_path=png_path,
            most_recommended_crop=most_recommended,
            translated_crop_name=translated_crop_name,
            rain_summary="Rain expected" if any(day['rain_yes'] for day in forecast_data) else "No rain expected",
            created_at=datetime.now()
        )
        db.session.add(prediction)
        db.session.commit()
        
        # Send email and SMS
        try:
            send_email(current_user, prediction, forecast_data, crop_recommendations, language)
            send_sms(current_user, prediction, language)
        except Exception as e:
            print(f"Error sending notifications: {str(e)}")
        
        # Store data in session for result page
        session['prediction_id'] = prediction.id
        session['forecast_data'] = forecast_data
        session['crop_recommendations'] = crop_recommendations
        session['most_recommended'] = most_recommended
        session['translated_crop_name'] = translated_crop_name
        
        return redirect(url_for('result'))
        
    return render_template('predict.html')

@app.route('/result')
@login_required
def result():
    prediction_id = session.get('prediction_id')
    if not prediction_id:
        return redirect(url_for('dashboard'))
        
    prediction = db.session.get(Prediction, prediction_id)
    forecast_data = session.get('forecast_data', [])
    crop_recommendations = session.get('crop_recommendations', [])
    most_recommended = session.get('most_recommended', '')
    translated_crop_name = session.get('translated_crop_name', '')
    
    return render_template(
        'result.html',
        prediction=prediction,
        forecast_data=forecast_data,
        crop_recommendations=crop_recommendations,
        most_recommended=most_recommended,
        translated_crop_name=translated_crop_name,
        crop_names=CROP_NAMES[prediction.language]
    )

@app.route('/predictions')
@login_required
def predictions():
    user_predictions = db.session.execute(
        db.select(Prediction).where(Prediction.user_id == current_user.id).order_by(Prediction.created_at.desc())
    ).scalars().all()
    return render_template('predictions.html', predictions=user_predictions)

@app.route('/prediction/<int:id>')
@login_required
def prediction_detail(id):
    prediction = db.session.get(Prediction, id)
    if prediction.user_id != current_user.id:
        flash('Unauthorized access', 'danger')
        return redirect(url_for('predictions'))
        
    return render_template('prediction_detail.html', prediction=prediction)

@app.route('/download/<path:filename>')
@login_required
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    flash('File not found', 'danger')
    return redirect(url_for('predictions'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)