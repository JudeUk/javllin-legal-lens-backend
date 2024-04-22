import os
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import pymongo
from rest_framework.permissions import AllowAny
from rest_framework.decorators import api_view, permission_classes
import json 
from django.http import JsonResponse
from langchain.chains import RetrievalQA
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

openAIEmbeddings = OpenAIEmbeddings()
llm = ChatOpenAI()



MONGODB_URI = os.getenv('DATABASE_URL')
client = pymongo.MongoClient(MONGODB_URI)
db = client.sample_mflix
constDb = client.ConstituitionDatabase
consituitionCollection =constDb.constituitionCollection

vectorstore = MongoDBAtlasVectorSearch(
    collection=consituitionCollection,
    embedding = openAIEmbeddings,  # text embedding function
    # embedding=huggingFaceEmbeddings,
    index_name="consSemanSearch",
)


@api_view(['POST'])
@csrf_exempt
@permission_classes([AllowAny])
def chat_constituition(request):
    try:
        body_data = json.loads(request.body.decode('utf-8'))
        query = body_data.get("query")
        
        retriever = vectorstore.as_retriever()
        qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
        
        # Run the retrieval QA
        retriever_output = qa.run(query)

        return JsonResponse({'message': retriever_output, 'as_output': "", 'retriever_output': retriever_output}, status=200)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    finally:
        client.close()