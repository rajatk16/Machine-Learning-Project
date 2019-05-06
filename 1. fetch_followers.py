# Import libraries
import tweepy
import time
import csv
import sys
import importlib

# Insert Twitter API KEYs and Access Tokens
consumer_key = 'vzmYMfTpVG2ZqGfWGqZwGvgdD'
consumer_secret = 'imGVfJ5S9Yy25ZXOwIoC1TM8e6piekiRfxSyIYkdQordxrHDmQ'
access_token = '76062506-ef6lvJMDBUwt1sWxqXRiuQy670kx3SFU2eQa2IoCH'
access_secret = 'cSIl4M98ZxjrJRshSpeoIbhOPW9lgUU2l9rP8LEdsO8hk'

# Insert the Twitter Handle
twitter_handle = 'react_india'

# Authorise the twitter API
auth = tweepy.auth.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth, 
wait_on_rate_limit=True, 
timeout=200,
wait_on_rate_limit_notify=True)

# Create a new CSV File
f = open('./followers.csv', 'w', newline='')
writer = csv.writer(f)
writer.writerow(['id', 'screen name', 'location'])

users = tweepy.Cursor(api.followers, screen_name=twitter_handle, count=200).items()
for u in users:
  user_id = u.id
  screen_name = u.screen_name
  location = u.location.encode('UTF-8')
  writer.writerow([user_id, screen_name, location])