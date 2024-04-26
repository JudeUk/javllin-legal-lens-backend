from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from pymongo import MongoClient
import json

# Assuming you've configured djongo properly to use MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client.sample_mflix
collection = db.cases

@csrf_exempt
def search_case(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        case_id = data.get('case_id', None)

        if case_id:
            # Searching for the case in MongoDB
            case = collection.find_one({'caseNumber': case_id})

            if case:
                # Case found, returning the case details
                response_data = {
                    'caseNumber': case.get('caseNumber'),
                    'caseTitle': case.get('caseTitle'),
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
