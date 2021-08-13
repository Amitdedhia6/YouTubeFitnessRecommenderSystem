from recommendation import RecommendationSystem
import numpy as np
from numpy import dot
from numpy.linalg import norm

# from tokenizer import Tokenizer
# t = Tokenizer()
# tokens = t.get_tokens('I bought twenty four apples in 2021')
from universal import nlp_global_object

# tags_text = "nodal officer"
# doc = nlp_global_object(tags_text)
# doc1 = nlp_global_object("bigger arms")
#
# count = 0
# vec = np.zeros(300)
# for token in doc:
#     vec += token.vector
#     count += 1
#
# avg_vec = vec / count
# count = 0
# vec = np.zeros(300)
# for token in doc1:
#     vec += token.vector
#     count += 1
#
# avg_vec1 = vec / count
#
# s = doc1.similarity(doc)
# print(s)
#
# cos_sim = dot(avg_vec, avg_vec1)/(norm(avg_vec)*norm(avg_vec1))
# # cos_sim = numpy.dot(doc.vector, doc1.vector) / (doc.vector* other.vector_norm)
#
# print(cos_sim)

spacy = RecommendationSystem()

while True:
    val = input("Enter search string ('quit' to exit the program): ")
    if val.lower() == 'quit':
        break

    recommendations = spacy.get_video_recommendations(val)
    if len(recommendations) == 0:
        print("No results...")
    else:
        sr_no = 0
        for video_id in recommendations:
            r = recommendations[video_id]
            sr_no += 1
            print(sr_no, ". ", r["title"], " --- ", r["url"])
            if sr_no >= 10:
                break
    print()
    print()
