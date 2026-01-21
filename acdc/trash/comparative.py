import json
import numpy as np

from datetime import timedelta


string_table = ''

# Link Case 0
with open(f"./acdc_link/cigre_eu_lv_acdc_link_case_0.json",'r') as fobj:
    data = json.loads(fobj.read())

td = timedelta(seconds=data['V_min']['t'])
string_table += f"{data['V_min']['value']:0.1f} ({data['V_min']['bus']},{td}) & "
string_table += f"{data['I_max_pu']['value_pu']:0.2f} ({data['I_max_pu']['line']},{timedelta(seconds=data['I_max_pu']['t'])})\n "

# Link Case 1
with open(f"./acdc_link/cigre_eu_lv_acdc_link_case_1.json",'r') as fobj:
    data = json.loads(fobj.read())

td = timedelta(seconds=data['V_min']['t'])
string_table += f"{data['V_min']['value']:0.1f} ({data['V_min']['bus']},{td}) & "
string_table += f"{data['I_max_pu']['value_pu']:0.2f} ({data['I_max_pu']['line']},{timedelta(seconds=data['I_max_pu']['t'])})\n "

# AC/DC 6w Case 0
with open(f"./acdc_6w/cigre_eu_lv_acdc_4w2w_case_0.json",'r') as fobj:
    data = json.loads(fobj.read())

td = timedelta(seconds=data['V_min']['t'])
string_table += f"{data['V_min']['value']:0.1f} ({data['V_min']['bus']},{td}) & "
string_table += f"{data['I_max_pu']['value_pu']:0.2f} ({data['I_max_pu']['line']},{timedelta(seconds=data['I_max_pu']['t'])})\n "

# AC/DC 6w Case 1
with open(f"./acdc_6w/cigre_eu_lv_acdc_4w2w_case_1.json",'r') as fobj:
    data = json.loads(fobj.read())
    print(data)

td = timedelta(seconds=data['V_min']['t'])
string_table += f"{data['V_min']['value']:0.1f} ({data['V_min']['bus']},{timedelta(seconds=data['V_min']['t'])}) & "
string_table += f"{data['I_max_pu']['value_pu']:0.2f} ({data['I_max_pu']['line']},{timedelta(seconds=data['I_max_pu']['t'])})\n "

# AC/DC 6w Case 2
with open(f"./acdc_6w/cigre_eu_lv_acdc_4w2w_case_2.json",'r') as fobj:
    data = json.loads(fobj.read())

td = timedelta(seconds=data['V_min']['t'])
string_table += f"{data['V_min']['value']:0.1f} ({data['V_min']['bus']},{timedelta(seconds=data['V_min']['t'])}) & "
string_table += f"{data['I_max_pu']['value_pu']:0.2f} ({data['I_max_pu']['line']},{timedelta(seconds=data['I_max_pu']['t'])})\n "


# AC/DC 4w Case 0
with open(f"./acdc_6w/cigre_eu_lv_acdc_4w2w_case_0.json",'r') as fobj:
    data = json.loads(fobj.read())

td = timedelta(seconds=data['V_min']['t'])
string_table += f"{data['V_min']['value']:0.1f} ({data['V_min']['bus']},{td}) & "
string_table += f"{data['I_max_pu']['value_pu']:0.2f} ({data['I_max_pu']['line']},{timedelta(seconds=data['I_max_pu']['t'])})\n "

# AC/DC 4w Case 1
with open(f"./acdc_6w/cigre_eu_lv_acdc_4w2w_case_1.json",'r') as fobj:
    data = json.loads(fobj.read())
    print(data)

td = timedelta(seconds=data['V_min']['t'])
string_table += f"{data['V_min']['value']:0.1f} ({data['V_min']['bus']},{timedelta(seconds=data['V_min']['t'])}) & "
string_table += f"{data['I_max_pu']['value_pu']:0.2f} ({data['I_max_pu']['line']},{timedelta(seconds=data['I_max_pu']['t'])})\n "

# AC/DC 4w Case 2
with open(f"./acdc_6w/cigre_eu_lv_acdc_4w2w_case_2.json",'r') as fobj:
    data = json.loads(fobj.read())

td = timedelta(seconds=data['V_min']['t'])
string_table += f"{data['V_min']['value']:0.1f} ({data['V_min']['bus']},{timedelta(seconds=data['V_min']['t'])}) & "
string_table += f"{data['I_max_pu']['value_pu']:0.2f} ({data['I_max_pu']['line']},{timedelta(seconds=data['I_max_pu']['t'])})\n "



print(string_table)