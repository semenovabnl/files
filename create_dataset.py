#!/usr/bin/env python
import numpy as np
import collections
import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from pprint import pprint
plt.figure(figsize=(10,15))
from os.path import isfile, join
import re

scopus_path =  os.getcwd()+'/scopus_files'
onlyfiles = [f for f in os.listdir(scopus_path) if isfile(join(scopus_path, f))]


class Person():
    def __init__(self):
        
        self.name = ""
        self.ID = 0
        self.neighbour_pubs = collections.OrderedDict()
        self.publications = collections.OrderedDict()
        self.neighbour = collections.OrderedDict()
        self.neighbour1 = collections.OrderedDict()
        self.neighbour3 = collections.OrderedDict()
        self.neighbour5 = collections.OrderedDict()
        self.numPub = 0
        self.numPub1 = 0
        self.numPub3 = 0
        self.numPub5 = 0

        self.numPubY = 0
        self.currYear = 2017
        self.predictYear = 2018
    
    def addNeigbour(self, neighbour, year, publication):
        if neighbour not in self.neighbour:
            self.neighbour[neighbour]=1
        else:
            self.neighbour[neighbour]+=1
            
        if publication not in self.publications:
            self.publications[publication]=1
        else:
            self.publications[publication]+=1
        
        if neighbour not in self.neighbour_pubs:
            self.neighbour_pubs[neighbour]=[]
            self.neighbour_pubs[neighbour].append(publication)
        else:
            self.neighbour_pubs[neighbour].append(publication)
            
        if year==self.currYear:
            if neighbour not in self.neighbour1:
                self.neighbour1[neighbour]=1
            else:
                self.neighbour1[neighbour]+=1

        if year>=2013 and year<=self.currYear:
            if neighbour not in self.neighbour3:
                self.neighbour3[neighbour]=1
            else:
                self.neighbour3[neighbour]+=1
        if year>=2011 and year<=self.currYear:
            if neighbour not in self.neighbour5:
                self.neighbour5[neighbour]=1
            else:
                self.neighbour5[neighbour]+=1                  
    def addPub(self, year):
       
            
        if(year<=self.currYear):
            self.numPub+=1
            if self.currYear-year==0:
                self.numPub1+=1 #lastyear
            if self.currYear-year<=2: #last 3 years: 2014,2015,2016
                self.numPub3+=1
            if self.currYear-year<=4: #last 5 years: 2012,2013,2014,2015,2016
                self.numPub5+=1
                
        if year >=self.predictYear:
            self.numPubY+=1
            
class Publication():
    def __init__(self):
        self.EID = 0
        self.pubType=""
        self.pubCite=''
        self.title=""
        self.journal=""
        self.year=0
        self.authors=""
        self.authorsID=""
        self.abstact=""
        self.authKW=""
        self.journalKW=""
        self.lang=""
        self.link=''
        
    #form a dict of journals
Publications = collections.OrderedDict()
i = 0

for f in onlyfiles:
    data = pd.read_csv(scopus_path+"/"+f, skiprows=1, header=None, 
                    names=["Authors", "AuthID", "Name", "Year", "Journal",  "Cite", 
                            "Link", "Abstract","AuthKW","JournalKW",
                           "Lang","Type","Source","EID"], 
                    converters={"Authors":str, "AuthID": str, "Name" : str, "Year" :int, "Journal" :str,  "Cite" :str, 
                            "Link" :str, "Abstract" :str,"AuthKW" :str,"JournalKW" : str, 
                           "Lang" :str,"Type" :str,"Source":str,"EID":str})
    
       
    Authors = data.Authors
    AuthID = data.AuthID
    Name = data.Name
    Year = data.Year
    Journal = data.Journal
    Cite = data.Cite
    Link = data.Link
    Abstract = data.Abstract
    AuthKW = data.AuthKW
    JournalKW = data.JournalKW
    Lang = data.Lang
    Type = data.Type
    Source = data.Source
    EID = data.EID.apply(lambda x: int(re.sub(r'2-s2.0-', '', x))) #число
    N = len(data.EID)
    for k in range(N):
#         if : #exists journal name and HSE authors
        pub = Publication()
        pub.EID = EID[k]
        pub.title = Name[k]
        pub.pubType = Type[k]
        pub.pubCite = Cite[k]
        pub.journal = Journal[k]
        pub.year = Year[i]
        pub.authors = Authors[k]
        pub.authorsID = AuthID[k]
        pub.abstact = Abstract[k]
        pub.authKW = AuthKW[k]
        pub.journalKW = JournalKW[k]
        pub.lang = Lang[k]
        pub.link = Link[k]


        Publications[EID[k]] = pub
    i+=1
    print(i)
    
AuthDict = collections.OrderedDict()

for I in range(len(Publications.keys())):
    k = list(Publications.keys())[I]
    curr_y = Publications[k].year
 

    Auth = re.sub(r'\,(?=\sJr[\.\,])', '', Publications[k].authors)
    Auth = re.sub(r' LHCb Collaboration|, GBD 2017 Disease and Injury Incidence and Prevalence Collaborators|, GBD 2017 SDG Collaborators', '', Auth)
    Auth = re.sub(r'GBD Tuberculosis Collaborators, |, GBD 2017 Risk Factor Collaborators', '', Auth)
    Auth = re.sub(r', GBD 2017 DALYs and HALE Collaborators|GBD 2017 Causes of Death Collaborators, ', '', Auth).split(", ")

    AuthID = Publications[k].authorsID.rstrip(';').split(';')
    N = len(AuthID) #number of HSE authors
    authList = []
    for a in range(N): #for each author
#         if len(Auth[a])==len(AuthID[a]):
        if(len(AuthID[a])>0):
            aa = AuthID[a] #translate
            authList.append(aa) #group of couthors

            if aa in AuthDict:
                AuthDict[aa].addPub(curr_y)
            else:
                AuthDict[aa] = Person()
#                 AuthDict[aa].name = aa # !!! было aa
                AuthDict[aa].ID = aa
    #                 AuthDict[aa].initial = Auth[a]
                AuthDict[aa].addPub(curr_y)

    for a in authList:
        for b in authList:
            if a!=b:
                AuthDict[a].addNeigbour(AuthDict[b], curr_y,Publications[k])
print('AuthDict done')

i = 0
AuthDict_names_dict = {list(AuthDict.keys())[i]: i for i in range(len(AuthDict.keys())) }

Graph_pub = nx.MultiGraph()
for k in AuthDict.keys():
    n1 = AuthDict[k]
    if len(n1.neighbour.keys())>0: #there are neighbours
        for n2 in list(n1.neighbour.keys()): #for each neighbour
            pubs = n1.neighbour_pubs[n2]
            for p in pubs:
                Graph_pub.add_edge(AuthDict_names_dict[n1.ID], AuthDict_names_dict[n2.ID],key = p.EID, year = p.year)#,p.ID])#{'year': p.year, 'ID' : p.ID})#  weight = p.
    else:
        Graph_pub.add_node(AuthDict_names_dict[n1.ID])

nx.write_gml(Graph_pub, "test.gml")
print('done')
raise SystemExit(1)