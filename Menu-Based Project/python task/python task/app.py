from flask import Flask, request, render_template, jsonify, send_file
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client
from googlesearch import search
import geocoder
from gtts import gTTS
import os
from pycaw.pycaw import AudioUtilities
import joblib

app = Flask(__name__)

# Function Definitions

def send_email(subject, body, to_address):
    smtp_server = 'smtp.example.com'
    smtp_port = 587
    smtp_user = 'your_email@example.com'
    smtp_password = 'your_password'

    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = to_address
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, to_address, msg.as_string())

def send_sms(to_phone, message_body):
    account_sid = 'your_account_sid'
    auth_token = 'your_auth_token'
    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body=message_body,
        from_='+1234567890',
        to=to_phone
    )
    return message.sid

def scrape_google(query):
    results = []
    for result in search(query, num_results=5):
        results.append(result)
    return results

def get_location():
    g = geocoder.ip('me')
    return g.latlng, g.address

def text_to_audio(text, filename='output.mp3'):
    tts = gTTS(text)
    tts.save(filename)
    return filename

def set_volume(volume_level):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        AudioUtilities.MMDeviceEnumerator, AudioUtilities.IAudioEndpointVolume, None)
    volume = interface.GetVolumeRange()[1]
    interface.SetMasterVolumeLevelScalar(volume_level / 100.0, None)

def send_bulk_emails(subject, body, addresses):
    for address in addresses:
        send_email(subject, body, address)

# Flask Routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_email', methods=['GET', 'POST'])
def handle_send_email():
    if request.method == 'POST':
        subject = request.form['subject']
        body = request.form['body']
        to_address = request.form['to_address']
        send_email(subject, body, to_address)
        return "Email sent"
    return render_template('send_email.html')

@app.route('/send_sms', methods=['GET', 'POST'])
def handle_send_sms():
    if request.method == 'POST':
        to_phone = request.form['phone_number']
        message_body = request.form['message']
        send_sms(to_phone, message_body)
        return "SMS sent"
    return render_template('send_sms.html')

@app.route('/scrape_google', methods=['GET', 'POST'])
def handle_scrape_google():
    if request.method == 'POST':
        query = request.form['query']
        results = scrape_google(query)
        return render_template('scrape_google.html', results=results)
    return render_template('scrape_google.html')

@app.route('/get_location', methods=['GET'])
def handle_get_location():
    latlng, address = get_location()
    return jsonify({'latlng': latlng, 'address': address})

@app.route('/text_to_audio', methods=['GET', 'POST'])
def handle_text_to_audio():
    if request.method == 'POST':
        text = request.form['text']
        filename = text_to_audio(text)
        return send_file(filename, as_attachment=True)
    return render_template('text_to_audio.html')

@app.route('/set_volume', methods=['GET', 'POST'])
def handle_set_volume():
    if request.method == 'POST':
        volume_level = float(request.form['volume_level'])
        set_volume(volume_level)
        return "Volume set to " + str(volume_level)
    return render_template('set_volume.html')

@app.route('/send_bulk_emails', methods=['GET', 'POST'])
def handle_send_bulk_emails():
    if request.method == 'POST':
        subject = request.form['subject']
        body = request.form['body']
        addresses = request.form['addresses'].split(',')
        send_bulk_emails(subject, body, addresses)
        return "Bulk emails sent"
    return render_template('send_bulk_emails.html')

if __name__ == "__main__":
    app.run(debug=True)
