from flask import Flask, render_template, request, redirect, url_for
import cv2
import smtplib
import pyttsx3
import boto3
from twilio.rest import Client
from googlesearch import search
from geopy.geocoders import Nominatim

app = Flask(__name__)

# Route for the Home page (Portfolio)
@app.route('/')
def index():
    return render_template('index.html')

# Route for About Me page
@app.route('/about')
def about():
    return render_template('about.html')

# Route for Projects page
@app.route('/projects')
def projects():
    return render_template('projects.html')

# Image Capture and Processing
@app.route('/capture_image')
def capture_image():
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    if ret:
        img_name = "static/captured_image.jpg"
        cv2.imwrite(img_name, frame)
    cam.release()
    return render_template('image.html', img_name=img_name)

# Google Search Results
@app.route('/google_search', methods=['GET', 'POST'])
def google_search():
    if request.method == 'POST':
        query = request.form['query']
        results = list(search(query, num_results=5))
        return render_template('search_results.html', query=query, results=results)
    return render_template('google_search.html')

# Get Current Location
@app.route('/get_location')
def get_location():
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.geocode("Your IP Address")
    return f"Location: {location.latitude}, {location.longitude}"

# Send Single Email
@app.route('/send_single_email', methods=['GET', 'POST'])
def send_single_email():
    if request.method == 'POST':
        sender = 'your_email@gmail.com'
        recipient = request.form['recipient']
        subject = request.form['subject']
        message = request.form['message']

        # Send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, 'your_password')
        server.sendmail(sender, recipient, f"Subject: {subject}\n\n{message}")
        server.quit()
        return "Email Sent!"
    return render_template('send_email.html')

# Send SMS with Twilio
@app.route('/send_sms', methods=['GET', 'POST'])
def send_sms():
    if request.method == 'POST':
        account_sid = 'your_account_sid'
        auth_token = 'your_auth_token'
        client = Client(account_sid, auth_token)

        message = client.messages.create(
            body="Hello from Flask!",
            from_='+1234567890',
            to=request.form['to']
        )
        return "SMS Sent!"
    return render_template('send_sms.html')

# Send Bulk Emails
@app.route('/send_bulk_email', methods=['GET', 'POST'])
def send_bulk_email():
    if request.method == 'POST':
        recipients = ['email1@gmail.com', 'email2@gmail.com']
        subject = request.form['subject']
        message = request.form['message']

        # Send bulk emails
        for recipient in recipients:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login('your_email@gmail.com', 'your_password')
            server.sendmail('your_email@gmail.com', recipient, f"Subject: {subject}\n\n{message}")
        server.quit()
        return "Bulk Emails Sent!"
    return render_template('send_bulk_email.html')

# Text-to-Audio Conversion
@app.route('/text_to_audio', methods=['GET', 'POST'])
def text_to_audio():
    if request.method == 'POST':
        text = request.form['text']
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return "Audio Generated!"
    return render_template('text_to_audio.html')

# Launch & Manage AWS EC2
@app.route('/manage_ec2', methods=['GET', 'POST'])
def manage_ec2():
    if request.method == 'POST':
        action = request.form['action']
        instance_id = request.form['instance_id']
        ec2 = boto3.client('ec2', region_name='your-region')

        if action == 'launch':
            ec2.run_instances(
                ImageId='ami-12345678', MinCount=1, MaxCount=1, InstanceType='t2.micro',
                KeyName='your-key-name'
            )
            return "EC2 Instance Launched!"
        elif action == 'stop':
            ec2.stop_instances(InstanceIds=[instance_id])
            return f"EC2 Instance {instance_id} Stopped!"
        elif action == 'terminate':
            ec2.terminate_instances(InstanceIds=[instance_id])
            return f"EC2 Instance {instance_id} Terminated!"
    return render_template('manage_ec2.html')

if __name__ == '__main__':
    app.run(debug=True)
