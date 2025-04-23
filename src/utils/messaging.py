import os
from twilio.rest import Client
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

def send_twilio_message():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(BASE_DIR, "keys", "twilio.txt")

    with open(filepath, 'r') as myfile:
        data = myfile.read()
    
    info_dict = eval(data)
    # Your Account SID from twilio.com/console
    account_sid = info_dict['account_sid']

    # Your Auth Token from twilio.com/console
    auth_token  = info_dict['auth_token']


    client = Client(account_sid, auth_token)

    message = client.messages.create( to = info_dict['your_num'], from_ = info_dict['trial_num'], body = "Intruder detected, you should check it out.")

def send_email(to_email):
    load_dotenv()
    from_email = os.getenv("EMAIL")
    password = os.getenv("PASSWORD")

    msg = MIMEText("Intruder Detected, you should check it out")
    msg['Subject'] = "Security Alert"
    msg['From'] = from_email
    msg['To'] = to_email

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())


def send_to_both():
    pass
