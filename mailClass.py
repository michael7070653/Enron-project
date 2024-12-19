import email
import email.utils
from typing import List
import pandas as pd
import re

from bs4 import BeautifulSoup

# from milstone import emails

"""
this class is used to extract the information from the email object
the class is used to extract the following information:
- the email text
- the email date
- the email content
- the email sender
- the email receiver
- the email day of the week
- the email day of the month
- the email month
- the email year
- the email hour
- the email subject
- the email message
- the email year and month
- the email string date
"""
class Enron_mail:
    # email_text:email.message_from_string
    # email_date = None
    # text = None
    def __init__(self, mail_text) -> None:
        ## Extracting the email text from the email object
        self.email_text = email.message_from_string(mail_text)
        ## Extracting the date of the email
        self.email_date = email.utils.parsedate_to_datetime(self.email_text["Date"])
        self.string_date =  (self.email_text["Date"])

        ## Extracting the content of the email
        self.text = self.get_text_from_email(self.email_text)
        self.text = self.text.split("-----Original Message-----")[0]
        self.text =  self.text.split("----------------------")[0]
    def get_content(self)->str:
        return self.text

    def get_from(self) -> str:
        return self.email_text["From"]

    def get_to(self) -> List[str]:
        return self.email_text["To"]


    def get_x_orig(self):
        return self.email_text["X-Origin"]


    def get_date(self):
        return self.email_date


    def get_day_week(self) -> int:
        x = self.email_date.weekday() + 2
        return x if x<= 7 else x % 7


    def get_day_month(self) -> int:
        return self.email_date.day

    def get_month(self) -> int:
        return self.email_date.month


    def get_string_date(self) -> str:
        return self.string_date

    def get_year(self) -> int:
        return self.email_date.year

    def get_hour(self) -> int:
        return self.email_date.hour

    def get_subject(self) -> str:
        return self.email_text["Subject"]


    def get_message(self):
        return self.email_text

    def get_year_month(self):
        return "-".join([str(self.email_date.year), str(self.email_date.month)])


    def get_text_from_email(self, msg):
        '''Extracts and returns the plain text content from email. Falls back to extracting HTML if plain text is not found.'''

        # Initialize a list to collect parts of the message content
        parts = []

        # Check if the email is multipart (meaning it contains multiple sections like plain text, HTML, attachments, etc.)
        if msg.is_multipart():
            # Walk through each part of the email
            for part in msg.walk():
                # If the part is plain text, append the payload to the list
                if part.get_content_type() == 'text/plain' and 'attachment' not in part.get('Content-Disposition', ''):
                    parts.append(
                        part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', errors='replace'))

                # Optionally, handle HTML content if no plain text is found
                elif part.get_content_type() == 'text/html' and 'attachment' not in part.get('Content-Disposition', ''):
                    # Use BeautifulSoup to extract text from the HTML part
                    html_content = part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8',
                                                                        errors='replace')
                    soup = BeautifulSoup(html_content, 'html.parser')
                    parts.append(soup.get_text())
        else:
            # If the email is not multipart, get the payload directly
            if msg.get_content_type() == 'text/plain':
                parts.append(
                    msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8', errors='replace'))
            elif msg.get_content_type() == 'text/html':
                # Fallback to extract text from HTML
                html_content = msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8',
                                                                   errors='replace')
                soup = BeautifulSoup(html_content, 'html.parser')
                parts.append(soup.get_text())

        # Join all parts and return the combined text
        return ''.join(parts).strip()



