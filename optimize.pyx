from __future__ import division

import cPickle
import numpy as np
import scipy.io as sio
import datetime 

class GenFeatures:
    
  def __init__(self):
    
    self.userIndex = cPickle.load(open("/users/chaitanya/PyCharmProjects/EventRec/Models/PE_userIndex.pkl", 'rb'))
    self.eventIndex = cPickle.load(open("/users/chaitanya/PyCharmProjects/EventRec/Models/PE_eventIndex.pkl", 'rb'))
    self.userEventScores = sio.mmread("/users/chaitanya/PyCharmProjects/EventRec/Models/PE_userEventScores").todense()
    self.userSimMatrix = sio.mmread("/users/chaitanya/PyCharmProjects/EventRec/Models/US_userSimMatrix").todense()
    self.eventPropSim = sio.mmread("/users/chaitanya/PyCharmProjects/EventRec/Models/EV_eventPropSim").todense()
    self.eventContSim = sio.mmread("/users/chaitanya/PyCharmProjects/EventRec/Models/EV_eventContSim").todense()
    self.numFriends = sio.mmread("/users/chaitanya/PyCharmProjects/EventRec/Models/UF_numFriends")
    self.userFriends = sio.mmread("/users/chaitanya/PyCharmProjects/EventRec/Models/UF_userFriends").toarray()
    self.eventPopularity = sio.mmread("/users/chaitanya/PyCharmProjects/EventRec/Models/EA_eventPopularity").todense()
    self.eventAttendees = cPickle.load(open("/users/chaitanya/PyCharmProjects/EventRec/Models/PE_eventAttendees.pkl", 'rb'))
    
    
  def userReco(self, userId, eventId):
    """
    for item i
      for every other user v that has a preference for i ( whether 1/-1)
        compute similarity s between u and v
        incorporate v's preference for i weighted by s into running aversge
    return top items ranked by weighted average
    """
    i = self.userIndex[userId]
    j = self.eventIndex[eventId]
    
    vs = self.userEventScores[:, j]  # All user scores for given eventId
    sims = self.userSimMatrix[i, :]  # User similarity scores for all users with given user
    
#     prod = sims * vs  
    prod = np.dot(sims,vs)
    
    # Multiply above matrices to get product score -> one float value, only those users who 
    # have some measure of similarity with the user(>0) and have rated the given event are counted in
    try:
      return prod[0, 0] - self.userEventScores[i, j]
    except IndexError:
      return 0


  def eventReco(self, userId, eventId):
    """
    for item i 
      for every item j that u has a preference for
        compute similarity s between i and j
        add u's preference for j weighted by s to a running average
    return top items, ranked by weighted average
    """
    i = self.userIndex[userId]
    j = self.eventIndex[eventId]
    
    js = self.userEventScores[i, :]
    psim = self.eventPropSim[:, j]
    csim = self.eventContSim[:, j]
    pprod = np.dot(js,psim)
    cprod = np.dot(js,csim)
#     pprod = js * psim
#     cprod = js * csim
    pscore = 0
    cscore = 0
    try:
      pscore = pprod[0, 0] - self.userEventScores[i, j]
    except IndexError:
      pass
    try:
      cscore = cprod[0, 0] - self.userEventScores[i, j]
    except IndexError:
      pass
    return pscore, cscore

  def userPop(self, userId):
    """
    Measures user popularity by number of friends a user has. People
    with more friends tend to be outgoing and are more likely to go
    to events
    """
    if self.userIndex.has_key(userId):
      i = self.userIndex[userId]
      try:
        return self.numFriends[0, i]
      except IndexError:
        return 0
    else:
      return 0


  def friendAttending(self,userId,eventId):
    
    i = self.userIndex[userId]
    j = self.eventIndex[eventId]
    
    fin = open("/users/chaitanya/PyCharmProjects/EventRec/data/user_friends.csv", 'rb')
    fin.readline()
    ln=0
    c=0
    score=0
    for line in fin:
      cols = line.strip().split(",")
      user = cols[0]
      if user==userId:
        
          friends = cols[1].split(" ")
          for friend in friends:
            if friend in self.eventAttendees[j][0]:  #yes
                score = score + 1
                c+=1
            if friend in self.eventAttendees[j][1]:  #no
                score = score - 1
                c+=1
            if friend in self.eventAttendees[j][2]:  #invited people = 0.5 times likelihood of attending the event
                score = score + 0.5
                c+=1
            
    if c!=0:
        return score/c
    else:
        return 0

  def friendInfluence(self, userId):
    """
    Measures friends influence by the friends who are known (from the
    training set) to go or not go to any event at all. The average of scores across
    all friends of the user is the influence score.
    """
    
    i = self.userIndex[userId]
    return sum(self.userFriends[i, :])   # Calculates sum of all user-friend scores for this user

  def eventPop(self, eventId):
    """
    Measures event popularity by the number attending and not attending.
    """
    i = self.eventIndex[eventId]
    return self.eventPopularity[i, 0] + 0.5*self.eventPopularity[i,1]  # return (yes-no) + 0.5*invited 


  def timeDiff(self,eventId, timestamp):
        
    fin = open("/users/chaitanya/PyCharmProjects/EventRec/data/events.csv", 'rb')
    ln = 0
    
    for line in fin:

      ln += 1  
      if ln < 2:
        continue
    
    cols = line.strip().split(",") 
    dttm = datetime.datetime.strptime(cols[2], "%Y-%m-%dT%H:%M:%S.%fZ") 
    diff = dttm - timestamp
    diff2 = str(diff).split(" ")[0]
    return diff2
#     diff2 = str(diff).replace(" days","")
#     return diff2


  def eventuserLocation(self,userId,eventId):
    
    eventCountry = ""
    eventCity = ""
    userLocation = ""
    
    fin1 = open("/users/chaitanya/PyCharmProjects/EventRec/data/events.csv", 'rb')
    ln = 0

    for line in fin1:

      ln += 1  
      if ln < 2:
        continue

      cols = line.strip().split(",") 
     
      if cols[0] == eventId:
        eventCountry = cols[6]
        eventCity = cols[3]
        break;

    fin2 = open("/users/chaitanya/PyCharmProjects/EventRec/data/users.csv", 'rb')
    ln = 0

    for line in fin2:

      ln += 1  
      if ln < 2:
        continue

      cols = line.strip().split(",")
      if cols[0] == userId:
        userLocation = cols[5]
        break;
    
    if (userLocation!="") and (eventCity!="" or eventCountry!=""):
        if eventCity in userLocation:       # user is in the city of the event
            return 1
        elif eventCountry in userLocation:  # user is in the country of the event
            return 0.5
        else:                               # If user is not in country of event, location feature=>0
            return 0
    
    return 0


  def getData(self):
    """
    Create new features based on various recommender scores. This
    is so we can figure out what weights to use for each recommender's
    scores.
    """

    fin = open("/users/chaitanya/PyCharmProjects/EventRec/data/train.csv", 'rb')
    fout = open("/users/chaitanya/PyCharmProjects/EventRec/newdata/train.csv", 'wb')
    # write output header
    
    feat_cols = ["invited", "user_reco", "event_p_reco","event_c_reco", "user_pop",
                 "friend_attend","friend_infl","event_pop","time_diff", "location_diff",
                  "interested","not_interested"]    # don't need userId and eventId any more
    
    fout.write(",".join(feat_cols) + "\n")
    
    ln = 0
    
    for line in fin:                 # iterate over all user-event pairs in train.csv
      ln += 1
    
      if ln < 2:
        continue
        
      cols = line.strip().split(",")
      userId = cols[0]
      eventId = cols[1]
      invited = cols[2]
        
      tid = cols[3].replace("+00:00", "")
      dttm = datetime.datetime.strptime(tid, "%Y-%m-%d %H:%M:%S.%f") 
      time_diff = self.timeDiff(eventId,dttm)  

#     print "%s:%d (userId, eventId)=(%s, %s)" % (fn, ln, userId, eventId)
    
      user_reco = self.userReco(userId, eventId)
      event_p_reco, event_c_reco = self.eventReco(userId, eventId)
      user_pop = self.userPop(userId)
      friend_attend = self.friendAttending(userId,eventId)
      friend_infl = self.friendInfluence(userId)
      event_pop = self.eventPop(eventId) 
      location_diff = self.eventuserLocation(userId,eventId)
      
      features = [invited,user_reco,event_p_reco,event_c_reco,user_pop,
                  friend_attend,friend_infl,event_pop,time_diff,location_diff]
      features.append(cols[4]) # interested
      features.append(cols[5]) # not_interested
        
      fout.write(",".join(map(lambda x: str(x), features)) + "\n")
    
    fin.close()
    fout.close()

# When running with cython, the actual class will be converted to a .so
# file, and the following code (along with the commented out import below)
# will need to be put into another .py and this should be run.

#import CRegressionData as rd

def main():
  feat = GenFeatures()
  print "Getting feature data..."
  feat.getData()

if __name__ == "__main__":
  main()