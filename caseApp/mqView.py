from io import BytesIO
import tempfile
import PyPDF2
import docx2txt
from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.response import Response
from .models import Case
from .serializers import CaseDataSerializer, CaseSerializer
from django import views
from rest_framework.permissions import AllowAny
from rest_framework.decorators import api_view, permission_classes, action
import pymongo
import requests
import numpy as np
import os
import pika
import json
from rest_framework.renderers import JSONRenderer


MONGODB_URI = os.getenv('DATABASE_URL')

client = pymongo.MongoClient(MONGODB_URI)

HUGGINGFACETOKEN = os.getenv('HUGGINGFACETOKEN')

HUGGINGFACETOKENWriteToken = "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"

API_URL = os.getenv('API_URL')
headers = {"Authorization": f"Bearer {HUGGINGFACETOKEN}"}


db = client.sample_mflix
collection =db.movies
collectionCases = db.cases




from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt  




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




# Function to generate embeddings for text
def generate_embeddings(text: str) -> list[float]:
    response = requests.post(API_URL, headers={"Authorization": f"Bearer {HUGGINGFACETOKENWriteToken}"}, json={"inputs": text})
    # embeddings = response.json()["outputs"][0]["embeddings"]
    return response.json()

# Function to compute the average embedding
def compute_average_embedding(embeddings):
    # Convert list of embeddings to numpy array
    embeddings_array = np.array(embeddings)
    # Compute mean along the first axis (axis=0) to get average embedding
    average_embedding = np.mean(embeddings_array, axis=0)
    return average_embedding.tolist()





# Your existing Django imports and settings

# RabbitMQ connection parameters
RABBITMQ_URL = os.getenv('RABBITMQ_URL')
QUEUE_NAME = 'upload_file_queue'

# Establish connection to RabbitMQ
connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
channel = connection.channel()

# Declare the queue
channel.queue_declare(queue=QUEUE_NAME)

@csrf_exempt
@permission_classes([AllowAny])
def upload_file(request):
    if request.method == 'POST' and request.FILES:
        # Your existing file handling code

        # Your existing processing code
        # print("Request body:", request.body)      # Print request body
        # print("Request method:", request.method)  # Print request method (e.g., GET, POST)
        # print("Request path:", request.path)      # Print request path (e.g., '/upload/')
        # print("Request GET parameters:", request.GET)  # Print GET parameters
        # print("Request POST parameters:", request.POST)  # Print POST parameters
        print("Request files:", request.FILES)    # Print uploaded files
        # print("Request headers:", request.headers)  # Print request headers
        # print("Request cookies:", request.COOKIES)  # Print cookies sent with the request
        # print("Request user:", request.user) 

        uploaded_files = request.FILES.getlist('facts of case')

        extracted_texts = []
        embeddings = []
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
                    elif file_extension == 'docx':
                        text = docx2txt.process(in_memory_file)
                    
                    # Check if text extraction was successful before generating embeddings
                    if text:
                        extracted_texts.append(text)
                        print(extracted_texts)

                        # Generate embeddings for the extracted text
                        embedding = generate_embeddings(text)  # Pass the extracted text here
                        print(embedding)
                        embeddings.append(embedding)
                    else:
                        raise ValueError("Text extraction failed for file {}".format(f.name))
                else:
                    raise ValueError("Unsupported file type: {}".format(file_extension))
            except Exception as e:
                response = JsonResponse({'error': 'Error extracting text from file: {}'.format(e)}, status=422)
                response['Access-Control-Allow-Origin'] = 'https://javallin-frontend.vercel.app'
                return response

        # Compute average embedding for each document
        average_embeddings = [compute_average_embedding(doc_embeddings) for doc_embeddings in embeddings]

        # Serialize data to be sent to the queue
        data_to_queue = {
            'extracted_texts': extracted_texts,
            'average_embeddings': average_embeddings,
            # Add any other data you want to send
        }

        print(data_to_queue)
        # Convert data to JSON
        message_body = json.dumps(data_to_queue)

        # Publish message to the queue
        channel.basic_publish(exchange='',
                              routing_key=QUEUE_NAME,
                              body=message_body)

        # Close connection to RabbitMQ
        connection.close()

        response = Response({'message': 'Text extraction request queued successfully'}, status=200)

        response.accepted_renderer = JSONRenderer()

        response.renderer_context = {'request': request}  # Set the renderer context explicitly


        response.accepted_media_type = 'application/json'  # Set the media type explicitly

        # return JsonResponse({'message': 'Text extracted successfully', 'texts': extracted_texts, 'case_numbers':len(case_numbers)}, status=200)



        return response
    
    response = Response({'error': 'No file provided'}, status=400)
    response.accepted_renderer = JSONRenderer()
    response.renderer_context = {'request': request}  # Set the renderer context explicitly

    response.accepted_media_type = 'application/json'  # Set the media type explicitly

    # return JsonResponse({'message': 'Text extracted successfully', 'texts': extracted_texts, 'case_numbers':len(case_numbers)}, status=200)

    return response
