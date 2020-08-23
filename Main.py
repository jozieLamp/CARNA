from MVDD import MVDD


def run():
    features = []
    mvd = MVDD(features)

    allData = ['Age', 'Gender', 'Race', 'Wt', 'BMI', 'InitialHospDays', 'TotalHospDays', 'NYHA', 'MLHFS', 'AF', 'AlchE', 'ANGP', 'AOREG', 'AOST', 'ARRH', 'CABG', 'CARREST', 'COPD', 'CVD', 'CYTOE', 'DEPR', 'DIAB', 'FAMILE', 'GOUT', 'HEPT', 'HTN', 'HYPERE', 'HTRANS', 'ICD', 'IDIOPE', 'ISCHD', 'ISCHEME', 'MALIG', 'MI', 'MTST', 'OTHUNE', 'PACE', 'PERIPAE', 'PMRG', 'PTCI', 'PTREG', 'PVD', 'RENALI', 'SMOKING', 'STERD', 'STROKE', 'SVT', 'TDP', 'TIA', 'VAHD', 'VALVUE', 'VF', 'SixFtWlk', 'VO2', 'ALB', 'ALT', 'AST', 'BUN', 'CRT', 'DIAL', 'HEC', 'HEM', 'PLA', 'POT', 'SOD', 'TALB', 'TOTP', 'WBC', 'ACE', 'BET', 'NIT', 'ANGIO', 'CINF', 'DIUR', 'AMR', 'ATE', 'BEN', 'BIS', 'BUM', 'CAND', 'CAP', 'CAR', 'DIGX', 'DIN', 'DOB', 'DOP', 'ENA', 'ETH', 'FOS', 'FUR', 'LIS', 'LOSA', 'MET', 'MIL', 'MON', 'NAT', 'NIG', 'NIP', 'OTHAA', 'OTHA', 'OTHB', 'OTHD', 'PRO', 'QUI', 'RAM', 'TOP', 'TOR', 'TRA', 'VALSA', 'EjF', 'BPDIAS', 'BPSYS', 'HR', 'PV', 'MAP', 'PP', 'PPP', 'PPRatio']

    hemo = ['RAP', 'PAS', 'PAD', 'PAMN', 'CWP', 'PCWPMod', 'PCWPA', 'PCWPMN', 'CO',
       'CI', 'SVRHemo', 'MIXED', 'BPSYS', 'BPDIAS', 'HRTRT', 'RATHemo', 'MAP',
       'MPAP', 'CPI', 'PP', 'PPP', 'PAPP', 'VR', 'RAT', 'PPRatio', 'Age',
       'EjF']

    mvd.generateRandomGraph(nodes=hemo, maxBranches=3)
    # mvd.repeatedStuff(nodes)


def main():
    run()


if __name__ == "__main__":
    main()