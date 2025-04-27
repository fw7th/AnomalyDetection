"""
messaging.py
Provides notification functionality for security alerts via SMS and email.
This module implements a messaging system that can send notifications through
Twilio SMS and email when security events like intruder detection occur.
Author: fw7th
Date: 2025-04-26
"""

import os
from twilio.rest import Client
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv


class messaging_system:
    """Security notification system that sends alerts via SMS and email.
    
    This class provides methods to send notifications through Twilio SMS service
    and email when security events are detected, such as intruder detection.
    
    Attributes:
        your_num (str): Recipient's phone number for SMS notifications.
        your_mail (str): Recipient's email address for email notifications.
    """
    
    def __init__(self, your_num=None, your_mail=None):
        """Initialize the messaging system with recipient contact information.
        
        Args:
            your_num (str, optional): Phone number to send SMS alerts to. Defaults to None.
            your_mail (str, optional): Email address to send email alerts to. Defaults to None.
        """
        self.your_num = your_num        
        self.your_mail = your_mail
        
    def send_twilio_message(self):
        """Send an SMS alert using Twilio API.
        
        Reads Twilio credentials from a configuration file and sends an SMS
        notification about intruder detection to the configured phone number.
        No message is sent if phone number is not configured.
        
        Returns:
            None
        """
        if self.your_num:
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            filepath = os.path.join(BASE_DIR, "config", "keys", "twilio.txt")
            with open(filepath, 'r') as myfile:
                data = myfile.read()
            
            info_dict = eval(data)
            # Your Account SID from twilio.com/console
            account_sid = info_dict['account_sid']
            # Your Auth Token from twilio.com/console
            auth_token = info_dict['auth_token']
            client = Client(account_sid, auth_token)
            message = client.messages.create(
                to=self.your_num,
                from_=info_dict['trial_num'],
                body="Intruder detected, you should check it out."
            )
            
    def send_email(self):
        """Send an email alert using SMTP.
        
        Reads email credentials from environment variables and sends an email
        notification about intruder detection to the configured email address.
        No email is sent if email address is not configured.
        
        Returns:
            None
        """
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
        """Send alerts via both SMS and email.
        
        Convenience method that calls both SMS and email sending methods.
        Messages will only be sent to channels where recipient information
        is configured.
        
        Returns:
            None
        """
        self.send_twilio_message()
        self.send_email()
