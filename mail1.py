#import RPi.GPIO as GPIO
from subprocess import call
import time
import os
import glob
import smtplib
import base64
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import sys

gmail_user = "iotdemopan@gmail.com"
gmail_pwd = "paniotdemo987@"
FROM = 'iotdemopan@gmail.com'
TO = ['praveenamoorthy30@gmail.com'] #must be a list

#IMAGE



msg = MIMEMultipart()
time.sleep(1)
msg['Subject'] ="SECURITY"

#BODY with 2 argument

#body=sys.argv[1]+sys.argv[2]
body="THIS IS FROM PANTECH SOLUTION REGARDING SECURITY BREACH"          
msg.attach(MIMEText(body,'plain'))
time.sleep(1)


###IMAGE
fp = open("hp.jpg", 'rb')   		
time.sleep(1)
img = MIMEImage(fp.read())
time.sleep(1)
fp.close()
time.sleep(1)
msg.attach(img)
time.sleep(1)


try:
        server = smtplib.SMTP("smtp.gmail.com", 587) #or port 465 doesn't seem to work!
        print ("smtp.gmail")
        server.ehlo()
        print ("ehlo")
        server.starttls()
        print ("starttls")
        server.login(gmail_user, gmail_pwd)
        print ("reading mail & password")
        server.sendmail(FROM, TO, msg.as_string())
        print ("from")
        server.close()
        print ('successfully sent the mail')
except:
        print ("failed to send mail")


