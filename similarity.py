# -------------------------------------------------------------------------
# AUTHOR: Hunter Swanson
# FILENAME: similarity.py
# SPECIFICATION: description of the program
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy,
#pandas, or other sklearn modules.
#You have to work here only with standard dictionaries, lists, and arrays

# Importing some Python libraries
import csv

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

documents = []

#reading the documents in a csv file
with open('cleaned_documents.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         documents.append (row)
         print(row)

#input("Press enter to continue...")
#Building the document-term matrix by using binary encoding.
#You must identify each distinct word in the collection without applying any transformations, using
# the spaces as your character delimiter.
#--> add your Python code here

#vestigial from a different approach, used for bug testing (# of words)
#docTermList = []    #creates the list of all terms
#for row in documents:
#    for value in row[1:]: #ignores row ID
#        for word in value.split(): #splits row into words
#            if word not in docTermList: #adds them to list if not present
#                docTermList.append(word)
#docTermList.sort()
#print(len(docTermList))  #total list of words, sorted


#strips row ID
listOfRows = []
for row in documents:
    for value in row[1:]: #ignore row ID
        listOfRows.append(value)

#print(listOfRows)
print("Number of documents:")
print(len(listOfRows))

#converts to document-term matrix
vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", binary=True) #default doesn't include 1 letter words like A, or I. default=r”(?u)\b\w\w+\b”, binary=True gives binary rather than count
vectorizer.fit(listOfRows)
docTermMatrix = vectorizer.transform(listOfRows)
#print(docTermMatrix)  #401 x 14404, 401 documents, 14404 words
#print(docTermMatrix[0])    #docTermMatrix[i] is the document



# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors
# --> Add your Python code here

greatestV1 = 0
greatestV2 = 0
greatestCosine = 0.0
cosine = 0.0
for a in range(len(listOfRows)-1):
    for b in range(a+1, len(listOfRows)):
        cosine= cosine_similarity(docTermMatrix[a],docTermMatrix[b])
        if cosine > greatestCosine:
            greatestV1 = a
            greatestV2 = b
            greatestCosine = cosine




# Print the highest cosine similarity following the information below
# The most similar documents are document 10 and document 100 with cosine similarity = x
# --> Add your Python code here
print("The most similar documents are document " + str(greatestV1) + " and " + str(greatestV2) + " with cosine similarity=" + str(greatestCosine)+".")


input("Press enter to continue...")