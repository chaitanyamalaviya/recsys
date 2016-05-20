# import pandas as pd
#
# def gettrain():
# #CSV to JSON Conversion
#
#     # read CSV file directly from a URL and save the result
#     train = pd.read_csv('/users/chaitanya/PyCharmProjects/EventRec/data/train.csv')
#
#     no_attendees={}
#
# def get_attendees():
#
#     attendees = pd.read_csv('/users/chaitanya/PyCharmProjects/EventRec/data/event_attendees.csv')
#
#     for i in range(1,attendees.size):
#         no_attendees[str(event_attendees[:i]['event'])] = event_attendees[:i]['yes'].count()
#
#     print(no_attendees['855842686'])
#     return;
#     #
#     #
#     #
#     # users = pd.read_csv('/users/chaitanya/PyCharmProjects/EventRec/data/users.csv')
#     # user_friends = pd.read_csv('/users/chaitanya/PyCharmProjects/EventRec/data/user_friends.csv')
#     # events = pd.read_csv('/users/chaitanya/PyCharmProjects/EventRec/data/events.csv')
#
# get_attendees()

from pymongo import MongoClient, ASCENDING
import numpy as np
import csv
client = MongoClient()
db = client.recommend


def getuser():

    csvfile = open('/users/chaitanya/PyCharmProjects/EventRec/data/users.csv', 'r')
    user_info = csv.DictReader(csvfile)

    attr= ["user_id","locale","birthyear","gender","joinedAt","location","timezone"]
    db.user.drop()

    for record in user_info:
        row={}
        for field in attr:
            row[field]=record[field]

        db.user.insert_one(row)

    for u in db.user.find({"user_id": "627175141"}):
        print type(u["location"])

def getevent():

    event_info = csv.DictReader(open('/users/chaitanya/PyCharmProjects/EventRec/data/events.csv', 'r'))
    attr= ["event_id","user_id","start_time","city","state","zip","country","lat","lng"]

    for record in event_info:
        row={}
        for field in attr:
            row[field]=record[field]

        db.event.insert_one(row)

    event_attendees_info = csv.DictReader(open('/users/chaitanya/PyCharmProjects/EventRec/data/event_attendees.csv', 'r'))
    attr = ["event","yes","maybe","invited","no"]

    for record in event_attendees_info:
        row={}
        for field in attr:
            row[field]=record[field]

        db.event_attendees.insert_one(row)

    print db.event_attendees.find_one({"user_id": "1764963771"})

def getUserFriends():

    user_friends = csv.DictReader(open('/users/chaitanya/PyCharmProjects/EventRec/data/user_friends.csv', 'r'))
    attr= ["user","friends"]


    for record in user_friends:
        db.user[record["user"]] = record["friends"]

    user_friends_info = csv.DictReader(open('/users/chaitanya/PyCharmProjects/EventRec/data/user_friends.csv', 'r'))
    attr= ["user","friends"]

    for record in user_friends_info:
        row={}
        for field in attr:
            row[field]=record[field]

        db.user_friends.insert_one(row)

    print db.user_friends.count()

getevent()