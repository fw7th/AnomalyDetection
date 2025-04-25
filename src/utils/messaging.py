import os
from twilio.rest import Client
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

class messaging_system:
    def __init__(self, your_num=None, your_mail=None):
        self.your_num = your_num        
        self.your_mail = your_mail

    def send_twilio_message(self):
        if self.your_num:
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

            message = client.messages.create( to = self.your_num, from_ = info_dict['trial_num'], body = "Intruder detected, you should check it out.")

    def send_email(self):
        if self.your_mail:
            load_dotenv()
            from_email = os.getenv("EMAIL")
            password = os.getenv("PASSWORD")

            msg = MIMEText("Intruder Detected, you should check it out")
            msg['Subject'] = "Security Alert"
            msg['From'] = from_email
            msg['To'] = self.your_mail

            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(from_email, password)
                server.sendmail(from_email, self.your_mail, msg.as_string())

    def send_to_both(self):
        self.send_twilio_message()
        self.send_email()
        
