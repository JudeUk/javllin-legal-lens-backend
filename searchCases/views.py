import os
from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from pymongo import MongoClient
import json

import pymongo

MONGODB_URI = os.getenv('DATABASE_URL')

client = pymongo.MongoClient(MONGODB_URI)
db = client.sample_mflix
collection = db.cases

@csrf_exempt
def search_case(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        # case_id = data.get('case_id', None)
        case_title = data.get('case_title', None)

        if case_title:
            # Searching for the case in MongoDB
            case = collection.find_one({'case_title': case_title})

            if case:
                # Case found, returning the case details
                response_data = {
                    'case_number': case.get('case_number'),
                    'case_title': case.get('case_title'),
                    'court': case.get('court'),
                    'date': case.get('date')
                }
                return JsonResponse(response_data)
            else:
                # Case not found
                return JsonResponse({'error': 'Case not found'}, status=404)
        else:
            # If case_id is not provided
            return JsonResponse({'error': 'Case ID is required'}, status=400)
    else:
        # If request method is not POST
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)



