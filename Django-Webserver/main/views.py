from django.shortcuts import render
# from .util.calculate import *
from .util.Main import *


# Create your views here.
def index(response):
	return render(response, "main/base.html", {})

def all(response):
	return render(response, "main/all.html", {})

def results(response):
	return render(response, "main/results.html", {})

def external(request):
	input = {}
	input['Age'] = request.POST.get('age')
	input['Gender'] = 1
	input['Race'] = 1
	input['EjF'] = request.POST.get('ef')
	input['HRTRT'] = request.POST.get('hr')
	input['BPDIAS'] = request.POST.get('dia')
	input['BPSYS'] = request.POST.get('sys')
	input['RAP'] = request.POST.get('rap')
	input['PAS'] = request.POST.get('pas')
	input['PAD'] = request.POST.get('pad')
	input['PAMN'] = request.POST.get('pamn')
	input['PCWP'] = request.POST.get('pcwp_mn')
	input['CO'] = request.POST.get('co')
	input['CI'] = request.POST.get('ci')
	input['MIXED'] = request.POST.get('mixed')
	input['MPAP'] = request.POST.get('mpap')
	input['MAP'] = request.POST.get('map')
	input['CPI'] = request.POST.get('cpi')
	input['PP'] = request.POST.get('pp')
	input['PPP'] = request.POST.get('ppp')
	input['PAPP'] = request.POST.get('papp')
	input['SVR'] = request.POST.get('svr')
	input['RAT'] = request.POST.get('rat')
	input['PPRatio'] = request.POST.get('pprat')
	input['PAPi'] = request.POST.get('papi')
	input['SAPi'] = request.POST.get('sapi')
	input['CPP'] = request.POST.get('cpp')
	input['PRAPRat'] = request.POST.get('praprat')

	
	# input['testparam'] = request.POST.get('testparam')
	# results = run([sys.executable, '/Users/jdano/Documents/HemoPheno/HemoPheno4HF/newsite/mysite/main/calculate.py'], shell=False, stdout=PIPE)
	string = ""
	score = 0
	path = ""

	print("I STIL WORKY----------------------------------------------------------------------------------------------------------------")

	#string, score, path, outcome = get_results(input, request.POST.get('testparam'))
	string, score, path, outcome = runHemo(input, request.POST.get('testparam'))
	chance = ""
	color = ""
	if score == 1:
		chance = "LOW"
		color = "green"
	elif score == 2:
		chance = "LOW INTERMEDIATE"
		color = "lime"
	elif score == 3:
		chance = "INTERMEDIATE"
		color = "yellow"
	elif score == 4:
		chance = "INTERMEDIATE HIGH"
		color = "orange"
	else:
		chance = "HIGH"
		color = "red"

	desc = "Given the inputted data, the algorithm has returned a score of " + str(score) + ", which means that the risk level is " + chance + ". This score indicates that the patient has a less than 10% chance of the outcome: " + str(outcome)
	print("WORKY WORKY WORKY----------------------------------------------------------------------------------------------------------------")
	print("RENDERING: ",path)

	# print(results)
	return render(request, "main/results.html", {"score":score, "desc":desc, "path":str(path), "chance":chance, "color":color})
