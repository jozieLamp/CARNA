 *****************************************************************
   $Author: shost001 $
   $Date: 2008/02/22 20:15:43 $
   $Source: /ct/escape/transfer/deidentified_NIH/RCS/formats.sas,v $

   Purpose:   This program creates a SAS format library in this directory
              for all ESCAPE deidentified datasets found under
              /ct/escape/transfer/deidentified_NIH/data

   Assumptions:  This program is run in SAS v8

   -------------------------------------------------------------
   Change History
   -------------------------------------------------------------
   $Log: formats.sas,v $
   Revision 1.1  2008/02/22 20:15:43  shost001
   Initial revision

*****************************************************************;

libname library "." ;

proc format library=library;

  value ESACIT
         0  = "None"
         1  = "Trace"
         2  = "Moderate"
         3  = "Massive"
         ;
  value ESARRH
         1  = "Sudden death with resuscitation"
         2  = "Supraventricular arrhythmia"
         3  = "Ventricular arrhythmia"
         4  = "ICD firing"
         5  = "AV block"
         6  = "Syncope"
         ;
  value ESCAR
         1  = "<1.8"
         2  = "1.8-2.2"
         3  = "2.3-2.5"
         4  = ">2.5"
         ;
  value ESCCOM
         1  = "PAC associated bleeding requiring surgical intervention"
         2  = "PAC associated bleeding requiring transfusion"
         3  = "PAC associated pulmonary emboli"
         4  = "PAC associated cannulation of carotid artery"
         5  = "PAC associated VT > 30 seconds or VF"
         6  = "PAC associated thrombosis of a blood vessel"
         7  = "PAC associated complete heart block requiring pacemaker"
         8  = "PAC associated perforation or rupture of pulmonary artery"
         9  = "PAC associated pneumothorax"
         10  = "PAC knotting"
         11  = "PAC associated valvular trauma"
         12  = "PAC associated infection"
         ;
  value ESCERT
         1  = "1"
         2  = "2"
         3  = "3"
         4  = "4"
         5  = "5"
         ;
  value ESCOM
         1  = "Ventricular tachycardia/fibrillation"
         2  = "Inappropriate firing"
         3  = "Cardiogenic shock (SBP < 60 mmHg requiring vasopressors)"
         4  = "Ischemia/Angina"
         5  = "Myocardial infarction"
         6  = "New atrial fibrillation/flutter"
         7  = "Pulmonary embolism"
         8  = "Stroke"
         9  = "TIA"
         10  = "Bradycardic arrest"
         11  = "Ventricular fibrillation"
         12  = "Ventricular tachycardia > 30 seconds"
         13  = "EMD/PEA"
         14  = "Undetermined cause"
         15  = "Other"
         16  = "Sepsis"
         17  = "Other infection requiring antibiotics"
         ;
  value ESCON
         1  = "Clinic visit"
         2  = "Telephone call"
         3  = "Rehospitalization"
         4  = "Continuous hospitalization since randomization"
         5  = "Lost to follow-up"
         98  = "Other"
         ;
  value ESCORE
         1  = "Cardiac transplantation"
         2  = "Consent withdrawn"
         3  = "Lost to follow-up"
         4  = "Protocol violation"
         5  = "Physician decision"
         6  = "Early study termination"
         ;
  value ESCPX
         1  = "Too critically ill to be taken out of bed and exercised"
         2  = "Unable to walk > 50 meters on the 6 minute walk."
         3  = "Patient cannot walk for technical reasons"
         4  = "Not done due to oversight."
         ;
  value ESCPXT
         1  = "Bicycle"
         2  = "Treadmill"
         ;
  value ESDISC
         1  = "Discharged home"
         2  = "Discharged to assisted living"
         3  = "Discharged to skilled nursing facility"
         ;
  value ESDIUR
         1  = "Renal dysfunction"
         2  = "Hypotension"
         98  = "Other"
         ;
  value ESDTH
         1  = "Pump failure"
         2  = "Fatal myocardial infarction"
         3  = "Unexpected  sudden death"
         4  = "Other cardiovascular"
         5  = "Cancer"
         6  = "Non-cardiovascular death"
         7  = "Unable to determine"
         ;
  value ESED
         1  = "Grade school"
         2  = "Technical school"
         3  = "Some high school"
         4  = "Undergraduate school"
         5  = "High school"
         6  = "Graduate school"
         ;
  value ESEDMA
         0  = "0"
         1  = "1+"
         2  = "2+"
         3  = "3+"
         4  = "4+"
         ;
  value ESEST
         1  = "0-25%"
         2  = "26-50%"
         3  = "51-75%"
         4  = "76-100%"
         ;
  value ESEXTR
         1  = "Cool"
         2  = "Lukewarm"
         3  = "Warm"
         ;
  value ESFQ
         1  = "QD"
         2  = "QOD"
         98  = "Other"
         ;
  value ESHEM
         1  = "Screening"
         2  = "Baseline 1"
         3  = "Baseline 2"
         4  = "8 am"
         5  = "4 pm"
         6  = "Optimal (lowest PCWP)"
         7  = "Standing"
         8  = "Supine"
         ;
  value ESHEPA
         0  = "Absent"
         1  = "2-4 finger breadths"
         2  = "> 4 finger breadths"
         ;
  value ESHF
         1  = "Alcoholic"
         2  = "Cytotoxic drug therapy"
         3  = "Familial"
         4  = "Hypertensive"
         5  = "Idiopathic"
         6  = "Ischemic"
         7  = "Peripartum"
         8  = "Valvular"
         9  = "Other/uncertain"
         ;
  value ESHIST
         1  = "Angina pectoris"
         2  = "Myocardial infarction (MI)"
         3  = "Percutaneous transluminal coronary intervention (PTCI)"
         4  = "Coronary artery bypass graft (CABG)"
         5  = "Primary tricuspid regurgitation"
         6  = "Mitral stenosis"
         7  = "Primary mitral regurgitation"
         8  = "Aortic stenosis"
         9  = "Aortic regurgitation"
         10  = "Peripheral vascular disease"
         11  = "Chronic obstructive pulmonary disease"
         12  = "Chronic steroid use"
         13  = "Diabetes"
         14  = "Insulin dependent"
         15  = "Controlled by oral agents"
         16  = "Gout"
         17  = "Hepatic disease"
         18  = "Hypertension"
         19  = "TIA"
         20  = "Stroke"
         21  = "Atrial fibrillation"
         22  = "Sustained ventricular tachycardia"
         23  = "Torsades de pointe"
         24  = "Ventricular fibrillation"
         25  = "Cardiac arrest/Rhythm unknown"
         26  = "Implantable cardiac defibrillator"
         27  = "Pacemaker placement"
         28  = "Malignancy"
         29  = "Depression (treated with prescription medications)"
         ;
  value ESHRT
         1  = "Ineligible"
         2  = "Active evaluation"
         3  = "No evaluation planned"
         ;
  value ESHSRE
         1  = "Heart failure exacerbation"
         2  = "Acute coronary syndrome"
         3  = "Other cardiovascular"
         4  = "Arrhythmia"
         5  = "Cancer"
         6  = "Non-cardiovascular"
         7  = "Unable to determine"
         ;
  value ESINC
         1  = "< 25,000"
         2  = "25-49,999"
         3  = "50-74,999"
         4  = "75-99,999"
         5  = ">=100,000"
         ;
  value ESINF
         1  = "Amrinone"
         2  = "Dobutamine"
         3  = "Dopamine"
         4  = "Milrinone"
         5  = "Nitroglycerin"
         6  = "Nitroprusside"
         7  = "Natracor"
         ;
  value ESINFT
         1  = "Pneumonia"
         2  = "UTI"
         3  = "Sepsis/Bacteremia"
         4  = "Cellulitis"
         5  = "IV line infection"
         98  = "Other (specify)"
         99  = "Unknown"
         ;
  value ESINHS
         1  = "3 Day"
         2  = "5 Day"
         3  = "7 Day"
         4  = "Optimal Day"
         ;
  value ESINS
         0  = "None"
         1  = "Private"
         2  = "Medicare"
         3  = "Medicaid"
         4  = "Private and Medicare"
         5  = "Medicare and Medicaid"
         ;
  value ESIV
         1  = "Amrinone"
         2  = "Dobutamine"
         3  = "Dopamine"
         4  = "Milrinone"
         5  = "Nitroglycerin"
         6  = "Nitroprusside"
         7  = "Natracor"
         ;
  value ESJVP
         0  = "Cannot measure"
         1  = "<8"
         2  = "8-12"
         3  = "12-16"
         4  = ">16"
         ;
  value ESLAB
         1  = "Hemoglobin (Hgb)"
         2  = "Platelets"
         3  = "Hematocrit (Hct)"
         4  = "WBC"
         5  = "Sodium"
         6  = "Potassium"
         7  = "BUN"
         8  = "Creatinine"
         9  = "ALT/SGPT"
         10  = "AST/SGOT"
         11  = "Total protein"
         12  = "Albumin"
         13  = "Total bilirubin"
         14  = "Direct bilirubin"
         ;
  value ESLUNT
         1  = "g/L"
         2  = "mmol/ L"
         3  = "g/dL"
         4  = "10 9/L OR 10 3/mm 3"
         5  = "/mm 3"
         6  = "L/L"
         7  = "%"
         8  = "mmol/L OR mEq/L"
         9  = "mg/dL"
         10  = "IU/L OR U/L OR mlU/mL"
         11  = "?mol/L"
         12  = "mcg/L OR ?g/L OR ng/mL"
         ;
  value ESLV
         1  = "Radionuclide ventriculogram"
         2  = "Ventricular angiography"
         3  = "Echocardiogram"
         ;
  value ESMDNM
         1  = "Benazepril"
         2  = "Captopril"
         3  = "Enalapril"
         4  = "Fosinopril"
         5  = "Lisinopril"
         6  = "Quinapril"
         7  = "Ramipril"
         8  = "Trandolapril"
         9  = "Other"
         10  = "Candesartan"
         11  = "Losartan"
         12  = "Valsartan"
         13  = "Other"
         14  = "Hydralazine"
         15  = "Isosorbide dinitrate"
         16  = "Isosorbide mononitrate"
         17  = "Topical nitroglycerin"
         ;
  value ESMED
         1  = "Statins"
         2  = "Other lipid lowering agents"
         3  = "Magnesium"
         4  = "Estrogen replacement therapy"
         5  = "Testosterone replacement therapy"
         6  = "Insulin"
         7  = "Oral diabetic agents"
         8  = "Aspirin (daily)"
         9  = "Other antiplatelet agents"
         10  = "NSAIDs"
         11  = "Thyroid replacement therapy"
         12  = "Antidepressants"
         13  = "Benzodiazepines"
         14  = "Allopurinol"
         15  = "Colchicine"
         16  = "Enoxaparin"
         17  = "Warfarin"
         18  = "Vitamin E"
         19  = "CoEnzyme Q10"
         20  = "Other antioxidants"
         21  = "Multi-vitamin"
         ;
  value ESMSR
         1  = "Needs only 1 pillow"
         2  = "Occasional orthopnea with 1 pillow"
         3  = "Needs 2 pillows most of the time"
         4  = "Needs 3 pillows most of the time"
         5  = "Needs 4 pillows most of the time (sitting up)"
         ;
  value ESNAME
         1  = "Bumetanide"
         2  = "Ethacrynic acid"
         3  = "Furosemide"
         4  = "Torsemide"
         5  = "Other"
         6  = "Amiloride"
         7  = "Spironolactone"
         8  = "Triamterene"
         9  = "Other"
         10  = "Chlorothiazide (diuril)"
         11  = "Hydrochlorothiazide (HCTZ)"
         12  = "Metolazone (zaroxolyn)"
         13  = "Other"
         ;
  value ESNOPA
         1  = "Abnormal anatomy"
         2  = "Other technical difficulties"
         3  = "Change in patient condition"
         4  = "Physician preference"
         5  = "Withdrawal of consent"
         6  = "Death"
         7  = "Other clinical event"
         ;
  value ESNOT
         1  = "Unlikely"
         2  = "Not related"
         ;
  value ESOTHC
         1  = "Procedure related"
         2  = "Stroke"
         3  = "Transient ischemic attack"
         4  = "Pulmonary embolism"
         98  = "Other"
         ;
  value ESPACN
         1  = "Randomized to CLIN"
         2  = "Randomized to PAC but catheter not placed due to"
         ;
  value ESPAS
         1  = "<40"
         2  = "40-50"
         3  = "51-60"
         4  = ">60"
         ;
  value ESPCWP
         1  = "PCWP A-wave (mean)"
         2  = "PWCP (mean of all waves)"
         3  = "PAD"
         ;
  value ESPER
         1  = "Baseline"
         2  = "Index Hospitalization"
         3  = "Discharge"
         4  = "2 Week Follow-up"
         5  = "1 Month Follow-up"
         6  = "2 Month Follow-up"
         7  = "3 Month Follow-up"
         8  = "6 Month Follow-up"
         99  = "Non visit specific forms"
         ;
  value ESPRO
         1  = "Dry/warm"
         2  = "Wet/warm"
         3  = "Dry/cold"
         4  = "Wet/cold"
         ;
  value ESPROC
         1  = "ICD implantation"
         2  = "CABG"
         3  = "Left heart catheterization"
         4  = "Cardiopulmonary resuscitation"
         5  = "Cardioversion"
         6  = "Intra-aortic balloon pump"
         7  = "Left ventricular assist device"
         8  = "Mechanical ventilation"
         9  = "PTCI"
         10  = "Permanent pacemaker"
         11  = "Temporary pacemaker"
         12  = "Other cardiac procedure/operation"
         13  = "Pulmonary Artery Catheterization"
         ;
  value ESRALE
         0  = "None"
         1  = "<1/3"
         2  = "1/3-2/3"
         3  = ">2/3"
         ;
  value ESRAP
         1  = "<8"
         2  = "8-12"
         3  = "13-16"
         4  = ">16"
         ;
  value ESRATE
         0  = "0"
         1  = "1"
         2  = "2"
         3  = "3"
         4  = "4"
         5  = "5"
         ;
  value ESRD
         1  = "Probably"
         2  = "Possibly"
         3  = "Not associated"
         4  = "Unable to determine"
         ;
  value ESREMV
         1  = "PAC associated complications"
         2  = "Technical problems"
         3  = "Change in patient condition"
         4  = "Death"
         98  = "Other"
         ;
  value ESREN
         0  = "Neither"
         1  = "History of creatinine > 3.5mg/dL"
         2  = "History of chronic dialysis"
         ;
  value ESRES
         1  = "Home"
         2  = "Assisted living"
         3  = "Skilled nursing facility"
         ;
  value ESRESP
         0  = "No"
         1  = "Yes"
         2  = "Unable to determine"
         ;
  value ESRESU
         1  = "Attempt cardiopulmonary resuscitation"
         2  = "Attempt cardiopulmonary resuscitation but do not intubate"
         3  = "Do not attempt cardiopulmonary resuscitation"
         ;
  value ESREV
         1  = "Cardiologist"
         2  = "Infectious Disease Specialist"
         3  = "Committee"
         ;
  value ESRHY
         1  = "Sinus bradycardia"
         2  = "Atrial fibrillation/flutter"
         3  = "Normal sinus rhythm"
         4  = "Paced rhythm"
         5  = "Sinus tachycardia"
         98  = "Other"
         ;
  value ESSER
         1  = "Death"
         2  = "Life Threatening"
         3  = "Persistent significant disability"
         4  = "Prolonged inpatient hospitalization"
         5  = "Inpatient medical complication that jeapordized the patient and required medical/surgical intervention"
         ;
  value ESSIGN
         1  = "+"
         2  = "-"
         ;
  value ESSMOK
         0  = "Never"
         1  = "Current"
         2  = "Quit < 6 months ago"
         3  = "Quit >= 6 months ago"
         ;
  value ESSTAT
         1  = "Ineligible"
         2  = "Active evaluation"
         3  = "Listed"
         4  = "Received transplant"
         5  = "Accepted but waiting to determine need after discharge"
         6  = "Not evaluated"
         ;
  value ESSUDN
         1  = "Identified arrhythmia"
         2  = "Witnessed cardiac arrest"
         3  = "Unwitnessed cardiac arrest"
         4  = "Sudden death associated with unexpected worsening of heart failure"
         ;
  value ESSYN
         1  = "Myocardial infarction"
         2  = "Unstable angina"
         3  = "Chest pain unspecified"
         ;
  value ESTERM
         1  = "Patient completed testing"
         2  = "Symptom limited"
         3  = "Angina"
         4  = "Serious arrhythmia"
         5  = "Blood pressure changes"
         6  = "No longer able to walk"
         98  = "Other"
         ;
  value ESTRO
         1  = "I"
         2  = "T"
         ;
  value ESVAL
         1  = "Normal"
         2  = "Absent overshoot"
         3  = "Square-wave"
         4  = "Uncertain"
         5  = "Not applicable"
         ;
  value ESWEDG
         1  = "<12"
         2  = "12-22"
         3  = "23-30"
         4  = ">30"
         ;
  value ESWHN
         0  = "None"
         1  = "Occasional"
         2  = "Constant"
         ;
  value ESWKUT
         1  = "Feet"
         2  = "Meters"
         ;
  value ESWLK
         1  = "The patient was too critically ill to be taken out of bed and exercised"
         2  = "Patient cannot walk for technical reasons"
         3  = "Not done due to oversight"
         ;
  value ESWSRT
         1  = "Abdominal discomfort"
         2  = "Breathing"
         3  = "Body swelling"
         4  = "Fatigue"
         ;
  value ZCLASS
         1  = "I"
         2  = "II"
         3  = "III"
         4  = "IV"
         ;
  value ZHGTU
         1  = "Inches"
         2  = "Centimeters"
         ;
  value ZMONTH
         1  = "JAN"
         2  = "FEB"
         3  = "MAR"
         4  = "APR"
         5  = "MAY"
         6  = "JUN"
         7  = "JUL"
         8  = "AUG"
         9  = "SEP"
         10  = "OCT"
         11  = "NOV"
         12  = "DEC"
         ;
  value ZPOSNE
         1  = "Positive"
         2  = "Negative"
         ;
  value ZRACE
         1  = "Caucasian"
         2  = "Black"
         3  = "Asian"
         4  = "Hispanic"
         5  = "Native American"
         98  = "Other"
         ;
  value ZSEX
         1  = "Male"
         2  = "Female"
         ;
  value ZSIGTY
         1  = "Investigator"
         2  = "Study coordinator"
         ;
  value ZTMPU
         1  = "C"
         2  = "F"
         ;
  value ZWGTU
         1  = "Pounds"
         2  = "Kilograms"
         ;
  value ZYES
         1  = "Yes"
         ;
  value ZYESNO
         0  = "No"
         1  = "Yes"
         ;
  value ZYNNA
         0  = "No"
         1  = "Yes"
         2  = "NA"
         ;

**************************************************************
**************************************************************
**** PAC FORMATS ;

  value ESASCI
         1  = "none"
         2  = "trace"
         3  = "moderate"
         4  = "massive"
         5  = "NA"
         ;
  value ESEDEM
         1  = "0"
         2  = "1+"
         3  = "2+"
         4  = "3+"
         5  = "4+"
         6  = "NA"
         ;
  value ESETIO
         1  = "Ischemic"
         2  = "Idiopathic"
         3  = "Alcoholic"
         4  = "Cytotoxic drug therapy"
         5  = "Familial"
         6  = "Hypertensive"
         7  = "Peripartum"
         8  = "Valvular"
         9  = "Other or uncertain"
         ;
  value ESETIOP
         1  = "Ischemic"
         2  = "Idiopathic"
         3  = "Alcoholic"
         4  = "Cytotoxic drug therapy"
         5  = "Familial"
         6  = "Hypertensive"
         7  = "Peripartum"
         8  = "Valvular"
         9  = "Other or uncertain"
         ;
  value ESHITE
         1  = "<8 cm"
         2  = "8-11 cm"
         3  = "12-16 cm"
         4  = ">16 cm"
         5  = "Cannot measure"
         ;
  value ESLAB
         1  = "Hemoglobin (Hgb)"
         2  = "Platelets"
         3  = "Hematocrit (Hct)"
         4  = "WBC"
         5  = "Sodium"
         6  = "Potassium"
         7  = "BUN"
         8  = "Creatinine"
         9  = "ALT/SGPT"
         10  = "AST/SGOT"
         11  = "Total protein"
         12  = "Albumin"
         13  = "Total bilirubin"
         14  = "Direct bilirubin"
         ;
  value ESLUNT
         1  = "g/L"
         2  = "mmol/ L"
         3  = "g/dL"
         4  = "10 9/L OR 10 3/mm 3"
         5  = "/mm 3"
         6  = "L/L"
         7  = "%"
         8  = "mmol/L OR mEq/L"
         9  = "mg/dL"
         10  = "IU/L OR U/L OR mlU/mL"
         11  = "?mol/L"
         12  = "mcg/L OR ?g/L OR ng/mL"
         ;
  value ESMANU
         1  = "Abbott"
         2  = "Arrow International"
         3  = "Baxter"
         4  = "Medtronic"
         5, 98  = "Other"
         ;
  value ESMANUP
         1  = "Abbott"
         2  = "Arrow International"
         3  = "Baxter"
         4  = "Medtronic"
      5,98  = "Other"
         ;
  value ESMEGA
         1  = "absent"
         2  = "2-4 cm"
         3  = ">4 cm"
         4  = "NA"
         ;
  value ESPER
         1  = "Baseline"
         2  = "Index Hospitalization"
         3  = "Discharge"
         4  = "2 Week Follow-up"
         5  = "1 Month Follow-up"
         6  = "2 Month Follow-up"
         7  = "3 Month Follow-up"
         8  = "6 Month Follow-up"
         99  = "Non visit specific forms"
         ;
  value ESPLOC
         1  = "Cardiac Catheterization Laboratory"
         2  = "Coronary Care Unit or Intensive Care Unit"
         3  = "Other"
         ;
  value ESRALS
         1  = "none"
         2  = "<1/3"
         3  = "1/3-2/3"
         4  = ">2/3"
         5  = "NA"
         ;
  value ESRESU
         1  = "Attempt cardiopulmonary resuscitation"
         2  = "Attempt cardiopulmonary resuscitation but do not intubate"
         3  = "Do not attempt cardiopulmonary resuscitation"
         ;
  value ESSTAT
         1  = "Ineligible"
         2  = "Active evaluation"
         3  = "Listed"
         4  = "Received transplant"
         5  = "Accepted but waiting to determine need after discharge"
         6  = "Not evaluated"
         ;
  value ESTEMP
         1  = "cool"
         2  = "lukewarm"
         3  = "warm"
         4  = "NA"
         ;
  value ESTIME
         1  = "0-24 hrs before PAC placement"
         2  = "12-24 hrs after PAC placement"
         ;
  value ZCLASS
         1  = "I"
         2  = "II"
         3  = "III"
         4  = "IV"
         ;
  value ZMONTH
         1  = "JAN"
         2  = "FEB"
         3  = "MAR"
         4  = "APR"
         5  = "MAY"
         6  = "JUN"
         7  = "JUL"
         8  = "AUG"
         9  = "SEP"
         10  = "OCT"
         11  = "NOV"
         12  = "DEC"
         ;
  value ZRACE
         1  = "Caucasian"
         2  = "Black"
         3  = "Asian"
         4  = "Hispanic"
         5  = "Native American"
         98  = "Other"
         ;
  value ZSEX
         1  = "Male"
         2  = "Female"
         ;
  value ZYES
         1  = "Yes"
         ;
  value ZYNNA
         0  = "No"
         1  = "Yes"
         2  = "NA"
         ;
  value reason
          1 = 'Instability likely to require PAC in next 24 hours'
          2 = 'Milrinone within last 48 hours'
          3 = 'Need to define hemodymanics to establish diagnosis of heart failure vs. other condtion'
          4 = 'Patient cannot return for follow up'
          5 = 'NYHA < Class IV'
          6 = 'No hospitalization for CHF within past year'
          7 = 'LVEF >= 30%'
          8 = 'Hx of CHF < 3 months'
          9 = 'No ACE inhibitors or diuretucs attempted within past 3 months'
         10 = 'SBP > 125 mmHg'
         11 = 'No Sx of elevated filling pressure (dyspnea at rest, abd. discomfort, severe anorexia, nausea 2d hepatosplanchnic congestion)'
         12 = 'No physical findings (JVD > 10mm above RA; square-wave valsalva response, hepatomegaly, ascites, or edema in absence of other obvious causes; rales > 1/3 lung fields)'
         13 = 'Unable to place PAC within next 12 hours'
         14 = 'Active transplant listing'
         15 = 'Mechanical ventilation present or anticipated'
         16 = 'IABP or LVAD present or anticipated'
         17 = 'Dopamine or Dobutamine infusion at a rate > 3 mcg/kg/min or at any rate for > 24 hours'
         18 = 'MI or cardiac surgery within past 6 weeks'
         19 = 'Current MI or ACS'
         20 = 'Moderate to severe aortic or mitral stenosis'
         21 = 'Planned coronary revascularizaton'
         22 = 'Primary pulmonary hypertension'
         23 = 'Pulmonary infarct within past month'
         24 = 'Current pneumothorax'
         25 = 'Creatinine > 3.5 mg/dL'
         26 = 'Temperature > 37.8 C'
         27 = 'WBC > 13,000 mm3'
         28 = 'Heart failure 2d to specific treatable cause (severe anemia, hypothyroidism, systemic infection)'
         29 = 'Life expectancy < 1 year'
         30 = 'Pregnant or lactating'
         31 = 'Childbearing potential without accepted birth control'
         32 = 'Less than 16 years of age'
         33 = ' '
      other = 'Other';


run;
