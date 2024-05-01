import os
from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from pymongo import MongoClient
import json

import pymongo

MONGODB_URI = os.getenv('DATABASE_URL')

client = pymongo.MongoClient(MONGODB_URI)
db = client.sample_mflix
collection = db.cases

@api_view(['POST'])
@csrf_exempt
@permission_classes([AllowAny])
def case_search(request):
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
                    'court': case.get('case_court'),
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



@csrf_exempt
def text_index_search_case(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        # case_id = data.get('case_id', None)
        case_title = data.get('case_title', None)

        if case_title:
            # Searching for the case in MongoDB
            cases =  collection.find({"$text": {"$search": case_title}})

            if cases:
                # case = dict(case)
                # Case found, returning the case details
                # response_data = {
                #     'case_number': case.get('case_number'),
                #     'case_title': case.get('case_title'),
                #     'court': case.get('court'),
                #     'date': case.get('date')
                # }
                # return JsonResponse(response_data)
                response_data = []
                for case in cases:
                  
                  case_title = case.get('case_title')
                  if case_title:
                    case['case_title'] = case['case_title'].replace('\xa0', ' ')

                    response_data.append({
                        'case_number': case.get('case_number'),
                        'case_title': case.get('case_title'),
                        'court': case.get('court'),
                        'date': case.get('date')
                    })




                print(response_data)
                return JsonResponse(response_data, safe=False)
            else:
                # Case not found
                return JsonResponse({'error': 'Case not found'}, status=404)
        else:
            # If case_id is not provided
            return JsonResponse({'error': 'Case ID is required'}, status=400)
    else:
        # If request method is not POST
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)





