from io import BytesIO
import json
import tempfile
import PyPDF2
import docx2txt
from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.response import Response
import torch
from .models import Case
from .serializers import CaseDataSerializer, CaseSerializer
from django import views
from rest_framework.permissions import AllowAny
from rest_framework.decorators import api_view, permission_classes, action
import pymongo
import requests
import numpy as np
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt 
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
# from transformers import TransformersTokenizer
from sentence_transformers import SentenceTransformer
from langchain_community.llms import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings,OpenAIEmbeddings


MONGODB_URI = os.getenv('DATABASE_URL')

client = pymongo.MongoClient(MONGODB_URI)

HUGGINGFACETOKEN = os.getenv('HUGGINGFACETOKEN')

HUGGINGFACETOKENWriteToken = "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"

API_URL = os.getenv('API_URL')

headers = {"Authorization": f"Bearer {HUGGINGFACETOKEN}"}





db = client.sample_mflix
collection =db.movies


constDb = client.ConstituitionDatabase
consituitionCollection =constDb.constituitionCollection


caseNumber =0
caseTitle = ""
court=""
date=""
similar_cases_response = []

huggingFaceEmbeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

openAIEmbeddings = OpenAIEmbeddings()

# vectorstore = MongoDBAtlasVectorSearch(
#     collection=consituitionCollection,
#     embedding = huggingFaceEmbeddings,  # Replace with your text embedding function
#     # embedding=huggingFaceEmbeddings,
#     # index_name="caseSemanticSearch",
# )


vectorstore = MongoDBAtlasVectorSearch(
    collection=consituitionCollection,
    embedding = openAIEmbeddings,  # text embedding function
    # embedding=huggingFaceEmbeddings,
    index_name="consSemanSearch",
)

@api_view(['POST'])
@permission_classes([AllowAny])
def create(request):
        case = Case.objects.create(
            request.data.get('facts') ,
            request.data.get('facts') ,
            request.data.get('facts')            
        )
        serializer = CaseDataSerializer(data=request.data)

        print(serializer)




class CaseViewSet(viewsets.ViewSet):
    def create(self, request):
        serializer = CaseSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)




# Function to generate huggingFaceEmbeddings for text
def generate_huggingFaceEmbeddings(text: str) -> list[float]:
    response = requests.post(API_URL, headers={"Authorization": f"Bearer {HUGGINGFACETOKENWriteToken}"}, json={"inputs": text})
    # huggingFaceEmbeddings = response.json()["outputs"][0]["huggingFaceEmbeddings"]
    return response.json()

# Function to compute the average embedding
def compute_average_embedding(huggingFaceEmbeddings):
    # Convert list of huggingFaceEmbeddings to numpy array
    huggingFaceEmbeddings_array = np.array(huggingFaceEmbeddings)
    # Compute mean along the first axis (axis=0) to get average embedding
    average_embedding = np.mean(huggingFaceEmbeddings_array, axis=0)
    return average_embedding.tolist()

# Modify the upload_file function
@csrf_exempt
@permission_classes([AllowAny])
def upload_file(request):
    if request.method == 'POST' and request.FILES:
        uploaded_files = request.FILES.getlist('facts of case')

        extracted_texts = []
        huggingFaceEmbeddings = []
        case_numbers = []

        for f in uploaded_files:
            file_extension = f.name.split('.')[-1].lower()

            try:
                print(f.content_type)
                if file_extension in ['pdf', 'docx']:
                    # In-memory file handling
                    file_content = f.read()
                    in_memory_file = BytesIO(file_content)

                    if file_extension == 'pdf':
                        pdf_reader = PyPDF2.PdfReader(in_memory_file)
                        text = '\n'.join([page.extract_text() for page in pdf_reader.pages])
                        extracted_texts.append(text)
                    elif file_extension == 'docx':
                        text = docx2txt.process(in_memory_file)
                        extracted_texts.append(text)
                    print(extracted_texts)    

                    
                    # Generate huggingFaceEmbeddings for the extracted text
                    embedding = generate_huggingFaceEmbeddings(extracted_texts)
                    print(embedding)
                    huggingFaceEmbeddings.append(embedding)
                    
                else:
                    raise ValueError("Unsupported file type: {}".format(file_extension))
            except Exception as e:
                response = JsonResponse({'error': 'Error extracting text from file: {}'.format(e)}, status=422)
                response['Access-Control-Allow-Origin'] = 'https://javallin-frontend.vercel.app'
                return response

        # Compute average embedding for each document
        average_huggingFaceEmbeddings = [compute_average_embedding(doc_huggingFaceEmbeddings) for doc_huggingFaceEmbeddings in huggingFaceEmbeddings]

        # Compare the uploaded document with documents in your database
        for average_embedding in average_huggingFaceEmbeddings:
            # You need to replace 'Space Movies' with the text you want to compare with
            # results =  collection.aggregate([
            #     {
            #         "$vectorSearch": {
            #             "queryVector": average_embedding,
            #             "path": "plot_huggingFaceEmbeddings",
            #             "numCandidates": 100,
            #             "limit": 4,
            #             "index": "PlotSemanticSearch",
            #         }
            #     }
            # ])

            resultsCases =  collectionCases.aggregate([
                {
                    "$vectorSearch": {
                        "queryVector": average_embedding,
                        "path": "average_embedding",
                        "numCandidates": 100,
                        "limit": 4,
                        "index": "caseSemanticSearch",
                    }
                }
            ])


            for doc in resultsCases:
                # print(doc["title"]

                caseNumber = doc["case_number"]
                caseTitle = doc["case_title"]
                court= doc["case_title"]
                date= doc["date"]
               

                data = {
                        "case_number": caseNumber,
                        "date": date,
                        "case_title": caseTitle,
                        "court": court,
                        }
                similar_cases_response.append(data)

                case_numbers.append(doc["case_number"])

                # print(doc["case_number"])
                print(doc)

        return JsonResponse({'message': 'Text extracted successfully', 'data': similar_cases_response, 'case_numbers':len(case_numbers)}, status=200)

    return JsonResponse({'error': 'No file provided'}, status=400)






# Function to generate huggingFaceEmbeddings for text
def generate_huggingFaceEmbeddings_query(request) -> list[float]:
    text = request.POST.get('query')
    response = requests.post(API_URL, headers={"Authorization": f"Bearer {HUGGINGFACETOKENWriteToken}"}, json={"inputs": text})
    # huggingFaceEmbeddings = response.json()["outputs"][0]["huggingFaceEmbeddings"]
    return response.json()




# model_name = "sentence-transformers/all-MiniLM-L6-v2"
# model = SentenceTransformer(model_name)
# embeddinggggg = model.encode()

# tokenizer = TransformersTokenizer.from_pretrained(model_name)

# def get_embedding(text):
#   encoded_text = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
#   with torch.no_grad():
#     embedding = model(encoded_text)[0][0]  # Extract first token embedding (CLS)
#   return embedding.cpu().detach().numpy()



@csrf_exempt
@permission_classes([AllowAny])
def chat_constituition(request):
    body_unicode = request.body.decode('utf-8')
    body_data = json.loads(body_unicode)
    query = body_data.get("query")
    
    # Perform vector search
    # docs = vectorstore.similarity_search(query, K=1)
    # as_output = docs[0].page_content if docs else "No information on your query"

    # Initialize LLM and retriever
    llm = ChatOpenAI()
    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
    
    # Run the retrieval QA
    retriever_output = qa.run(query)

    # Return response
    return JsonResponse({'message': retriever_output, 'as_output': "", 'retriever_output': retriever_output}, status=200)