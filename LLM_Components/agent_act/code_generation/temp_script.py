from flask import Flask, request, render_template, redirect, url_for, flash
from flask_mail import Mail, Message
import os
import secrets
import hashlib
import time

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key')  # Use an environment variable

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.example.com'  # Replace with your mail server
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')  # Use an environment variable
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')  # Use an environment variable
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER')  # Use an environment variable

mail = Mail(app)

# Simulated user database
users = {
    'user@example.com': {'password': 'hashed_password', 'reset_token': None, 'token_time': None}
}

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()  # Hash the password

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        if email in users:
            # Generate a reset token
            token = secrets.token_urlsafe()
            users[email]['reset_token'] = token
            users[email]['token_time'] = time.time()  # Store the time of token generation
            
            # Create a reset link
            reset_link = url_for('reset_password', token=token, _external=True)
            
            # Send email
            msg = Message('Password Reset Request', recipients=[email])
            msg.body = f'Click the link to reset your password: {reset_link}'
            mail.send(msg)
            
            flash('A password reset link has been sent to your email.', 'success')
            return redirect(url_for('forgot_password'))
        else:
            flash('Email not found.', 'error')
    
    return render_template('forgot_password.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    email = None
    for user_email, user_data in users.items():
        if user_data['reset_token'] == token:
            email = user_email
            break
    
    if email is None or (time.time() - users[email]['token_time'] > 3600):  # Token expires after 1 hour
        flash('Invalid or expired token.', 'error')
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        new_password = request.form['password']
        # Hash the new password before saving it
        users[email]['password'] = hash_password(new_password)
        users[email]['reset_token'] = None  # Invalidate the token
        users[email]['token_time'] = None  # Clear the token time
        flash('Your password has been reset!', 'success')
        return redirect(url_for('login'))  # Redirect to login page
    
    return render_template('reset_password.html', token=token)

@app.route('/login')
def login():
    return 'Login Page'  # Replace with your login page

if __name__ == '__main__':
    app.run(debug=True)