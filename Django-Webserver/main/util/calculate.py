# from flask import Flask, redirect, url_for, render_template, request, Response
# app = Flask (__name__)
from . import Main


# @app.route('/',methods=["POST","GET"])
# def renderMain():
#     if request.method == "POST":

#         #SETTING VARS PASSED EMPTY TO NONE
#         hemoDict = {'RAP': [0.0, 85.0],
#             'PAS': [0.0, 90.0],
#             'PAD': [0.0, 59.0],
#             'PAMN': [0.0, 82.0],
#             'PCWP': [0.0, 53.0],
#             'PCWPMod': [0.0, 53.0],
#             'PCWPA': [0.0, 53.0],
#             'PCWPMN': [0.0, 49.0],
#             'CO': [0.0, 25.0],
#             'CI': [0.0, 4.81],
#             'SVRHemo': [0.0, 6866.0],
#             'MIXED': [0.0, 627.0],
#             'BPSYS': [0.0, 168.0],
#             'BPDIAS': [0.0, 125.0],
#             'HRTRT': [0.0, 125.0],
#             'RATHemo': [0.0, 3.74],
#             'MAP': [0.0, 226.0],
#             'MPAP': [0.0, 129.3333333],
#             'CPI': [0.0, 1.820192166],
#             'PP': [-55.0, 106.0],
#             'PPP': [-0.833333333, 0.736842105],
#             'PAPP': [0.0, 0.8444444440000001],
#             'SVR': [0.0, 10755.555559999999],
#             'RAT': [0.0, 3.736842105],
#             'PPRatio': [-0.8870967740000001, 9.666666667000001],
#             'Age': [23.0, 88.0],
#             'EjF': [0.0, 45.0]}

#         for key in hemoDict:
#             hemoDict[key] = request.form[key.lower()]

#         results = runHemo(hemoDict,request.form["testparam"])
#         stringy = results[0]



#         # return render_template("main/results.html",imgsrc=("."+stringy[8:]),score=results[1],path=results[2])
#         return results

#     return render_template("main/base.html")
"""
def get_results(data, testparam):
    hemoDict = {'RAP': [0.0, 85.0],
            'PAS': [0.0, 90.0],
            'PAD': [0.0, 59.0],
            'PAMN': [0.0, 82.0],
            # 'PCWP': [0.0, 53.0],
            # 'PCWPMod': [0.0, 53.0],
            # 'PCWPA': [0.0, 53.0],
            'PCWPMN': [0.0, 49.0],
            'CO': [0.0, 25.0],
            'CI': [0.0, 4.81],
            # 'SVRHemo': [0.0, 6866.0],
            'MIXED': [0.0, 627.0],
            'BPSYS': [0.0, 168.0],
            'BPDIAS': [0.0, 125.0],
            # 'HRTRT': [0.0, 125.0],
            # 'RATHemo': [0.0, 3.74],
            # 'MAP': [0.0, 226.0],
            'MPAP': [0.0, 129.3333333],
            'CPI': [0.0, 1.820192166],
            'PP': [-55.0, 106.0],
            'PPP': [-0.833333333, 0.736842105],
            'PAPP': [0.0, 0.8444444440000001],
            'SVR': [0.0, 10755.555559999999],
            'RAT': [0.0, 3.736842105],
            'PPRatio': [-0.8870967740000001, 9.666666667000001],
            'Age': [23.0, 88.0],
            'EjF': [0.0, 45.0]}

    for key in hemoDict:
        hemoDict[key] = data[key.lower()]

    #Josie added test code just to make it work
    hemoDict['PCWP'] = 36
    hemoDict['PCWPMod'] = 36
    hemoDict['HRTRT'] = 25
    hemoDict['PCWPA'] = 36
    hemoDict['SVRHemo'] = 30
    hemoDict['HRTRT'] = 80
    hemoDict['RATHemo'] = 30
    hemoDict['MAP'] = 58

    # results = runHemo(hemoDict, params['testparam'])
    results = Main.runHemo(hemoDict,testparam)
    print(hemoDict)
    print("RESULTS:", results)
    stringy = results[0]
    score = results[1]
    path = results[2]
    print(stringy, score, path)


    # return render_template("main/results.html",imgsrc=("."+stringy[8:]),score=results[1],path=results[2])
    #     return results

    # return request.form['pas']
    return stringy, score, path, testparam

"""
# if __name__ == "__main__":
#     app.run(debug=True)
