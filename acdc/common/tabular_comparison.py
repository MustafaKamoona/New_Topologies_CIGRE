import json

results_comparison = {
    'cigre_eu_lv_acdc_link_case_0':{'folder':'acdc_link'},
    'cigre_eu_lv_acdc_link_case_1':{'folder':'acdc_link'}
}


string = ''
for item in results_comparison:
    with open(f"../{results_comparison[item]['folder']}/{item}.json") as fobj:
        results = json.loads(fobj.read())

    
    string += f"{results['V_min']['value']:0.1f}\n"

print(string)
