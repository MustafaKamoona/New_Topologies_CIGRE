# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 16:28:03 2022

@author: jmmau
"""

import numpy as np
import pandas as pd


class loads:
    
    '''
    Case 0: original 
    Case 1: only AC loads + AC chargers
    Case 2: AC and DC loads and chargers, total power as in Case 1
    Case 3: AC single phase loads, total power as in Case 2
    '''
    
    def __init__(self):
        
        self.case = 'Case 0'
        
        #self.params = params
        self.xy_0 = False
        
        self.profiles = {}
        
        self.buses_ac = {}
        self.buses_ac['res'] = ['R01','R11','R15','R16','R17','R18']
        self.buses_ac['ind'] = ['I02']
        self.buses_ac['com'] = ['C01','C12','C13','C14','C17','C18','C19','C20'] 
        
        self.buses_dc = {}
        self.buses_dc['res'] = ['S11','S15','S16','S17','S18']
        self.buses_dc['ind'] = ['H02']
        self.buses_dc['com'] = ['D12','D17','D19','D20'] 
        
        self.totals = {'res':{},'ind':{},'com':{}}
        
    def set_loads(self,h,params):
        
        self.params = params

        if self.xy_0: 
            load_factor= 0.0 
        else: 
            load_factor = 1.0
                        
        
        for area in self.profiles.keys():

            self.factor   = np.interp(h,self.profiles[area]['time'],self.profiles[area]['factor'])   
            self.chargers = np.interp(h,self.profiles[area]['time'],self.profiles[area]['chargers']) 
            
            if self.case == 'Case 0': self.chargers = 0*self.chargers
                
            self.totals[area]['p_load_ac'] = 0.0
            self.totals[area]['p_charger_ac'] = 0.0
            self.totals[area]['q_load_ac'] = 0.0
            self.totals[area]['q_charger_ac'] = 0.0
            self.totals[area]['p_load_dc'] = 0.0
            self.totals[area]['p_charger_dc'] = 0.0            
            
            for item in self.buses_ac[area]:

                p_kw   = self.df_loads_ac.loc[item]['P (kW)']
                q_kvar = self.df_loads_ac.loc[item]['Q (kvar)']
                
                self.totals[area]['p_load_ac'] += self.factor*p_kw*1e3
                self.totals[area]['p_charger_ac'] += self.chargers*p_kw*1e3
                self.totals[area]['q_load_ac'] += self.factor*q_kvar*1e3
                self.totals[area]['q_charger_ac'] += self.chargers*q_kvar*1e3
                
                if self.case != 'Case 3' or item in ['R01','C01']:
                    for ph in ['a','b','c']:
                        self.params.update({f'p_load_{item}_{ph}':(self.factor+self.chargers)*1e3*p_kw/3})
                        self.params.update({f'q_load_{item}_{ph}':(self.factor+self.chargers)*1e3*q_kvar/3})
                else:
                    for ph in ['a']:
                        self.params.update({f'p_load_{item}_{ph}':(self.factor+self.chargers)*1e3*p_kw})
                        self.params.update({f'q_load_{item}_{ph}':(self.factor+self.chargers)*1e3*q_kvar})
                    for ph in ['b','c']:
                        self.params.update({f'p_load_{item}_{ph}':0.0})
                        self.params.update({f'q_load_{item}_{ph}':0.0})                 
                        
            for item in self.buses_dc[area]:

                p_kw   = self.df_loads_dc.loc[item]['Pdc (kW)']
                
                self.totals[area]['p_load_dc'] += self.factor*p_kw*1e3
                self.totals[area]['p_charger_dc'] += self.chargers*p_kw*1e3


                #self.params.update({f'p_load_{item}':(self.factor+self.chargers)*1e3*p_kw})          

    def read(self,excel_file):
        
        self.excel_file = excel_file
        
        self.df_loads_ac = pd.read_excel(self.excel_file, 
                      sheet_name=self.case, 
                      header=1, names=None, index_col=0, 
                      usecols=[1,4,5], 
                      skiprows=[8,9,10,12,13,14,23,24,25,26])

        self.df_loads_dc = pd.read_excel(self.excel_file, 
                      sheet_name=self.case, 
                      header=1, names=None, index_col=0, 
                      usecols=[7,8,10], 
                      skiprows=[2,8,9,10,12,13,14,15,17,18,20,23,24,25,26])
        
        df_profiles_res = pd.read_excel(self.excel_file, 
                      sheet_name='Profiles', 
                      header=4, names=None, index_col=0, 
                      usecols=[0,1,2], 
                      )

        df_profiles_ind = pd.read_excel(self.excel_file, 
                      sheet_name='Profiles', 
                      header=4, names=None, index_col=0, 
                      usecols=[3,4,5], 
                      )

        df_profiles_com = pd.read_excel(excel_file, 
                      sheet_name='Profiles', 
                      header=4, names=None, index_col=0, 
                      usecols=[6,7,8], 
                      )
        
        self.profiles['res'] = {}
        self.profiles['res']['time']     = df_profiles_res.index.values.astype(np.float64)
        self.profiles['res']['factor']   = df_profiles_res.values[:,0]/100
        self.profiles['res']['chargers'] = df_profiles_res.values[:,1]/100
        self.profiles['res']['p_kW'] = df_profiles_res.values[:,1]/100

        self.profiles['ind'] = {}
        self.profiles['ind']['time']     = df_profiles_ind.index.values.astype(np.float64)
        self.profiles['ind']['factor']   = df_profiles_ind.values[:,0]/100
        self.profiles['ind']['chargers'] = df_profiles_ind.values[:,1]/100

        self.profiles['com'] = {}
        self.profiles['com']['time']     = df_profiles_com.index.values.astype(np.float64)
        self.profiles['com']['factor']   = df_profiles_com.values[:,0]/100
        self.profiles['com']['chargers'] = df_profiles_com.values[:,1]/100

    def report_profiles(self,times):
        
        p_loads_res_ac,p_loads_ind_ac,p_loads_com_ac = [],[],[]
        p_chargers_res_ac,p_chargers_ind_ac,p_chargers_com_ac = [],[],[]

        q_loads_res_ac,q_loads_ind_ac,q_loads_com_ac = [],[],[]
        q_chargers_res_ac,q_chargers_ind_ac,q_chargers_com_ac = [],[],[]
        
        p_loads_res_dc,p_loads_ind_dc,p_loads_com_dc = [],[],[]
        p_chargers_res_dc,p_chargers_ind_dc,p_chargers_com_dc = [],[],[]
        
        for t in times:
            self.set_loads(t/3600,self.params)
            p_loads_res_ac += [self.totals['res']['p_load_ac']]
            p_loads_ind_ac += [self.totals['ind']['p_load_ac']]
            p_loads_com_ac += [self.totals['com']['p_load_ac']]
            p_chargers_res_ac += [self.totals['res']['p_charger_ac']]
            p_chargers_ind_ac += [self.totals['ind']['p_charger_ac']]
            p_chargers_com_ac += [self.totals['com']['p_charger_ac']]
            
            q_loads_res_ac    += [self.totals['res']['q_load_ac']]
            q_loads_ind_ac    += [self.totals['ind']['q_load_ac']]
            q_loads_com_ac    += [self.totals['com']['q_load_ac']]
            q_chargers_res_ac += [self.totals['res']['q_charger_ac']]
            q_chargers_ind_ac += [self.totals['ind']['q_charger_ac']]
            q_chargers_com_ac += [self.totals['com']['q_charger_ac']]

            p_loads_res_dc += [self.totals['res']['p_load_dc']]
            p_loads_ind_dc += [self.totals['ind']['p_load_dc']]
            p_loads_com_dc += [self.totals['com']['p_load_dc']]
            p_chargers_res_dc += [self.totals['res']['p_charger_dc']]
            p_chargers_ind_dc += [self.totals['ind']['p_charger_dc']]
            p_chargers_com_dc += [self.totals['com']['p_charger_dc']]
            #print()
            
        self.profiles['res']['p_load_ac'] = np.array(p_loads_res_ac)
        self.profiles['ind']['p_load_ac'] = np.array(p_loads_ind_ac)
        self.profiles['com']['p_load_ac'] = np.array(p_loads_com_ac)
        self.profiles['res']['p_charger_ac'] = np.array(p_chargers_res_ac)
        self.profiles['ind']['p_charger_ac'] = np.array(p_chargers_ind_ac)
        self.profiles['com']['p_charger_ac'] = np.array(p_chargers_com_ac)

        self.profiles['res']['q_load_ac']    = np.array(q_loads_res_ac)
        self.profiles['ind']['q_load_ac']    = np.array(q_loads_ind_ac)
        self.profiles['com']['q_load_ac']    = np.array(q_loads_com_ac)
        self.profiles['res']['q_charger_ac'] = np.array(q_chargers_res_ac)
        self.profiles['ind']['q_charger_ac'] = np.array(q_chargers_ind_ac)
        self.profiles['com']['q_charger_ac'] = np.array(q_chargers_com_ac)

        self.profiles['res']['s_load_ac']    = (np.array(q_loads_res_ac)**2 + np.array(q_loads_res_ac)**2)**0.5
        self.profiles['ind']['s_load_ac']    = (np.array(q_loads_ind_ac)**2 + np.array(q_loads_ind_ac)**2)**0.5
        self.profiles['com']['s_load_ac']    = (np.array(q_loads_com_ac)**2 + np.array(q_loads_com_ac)**2)**0.5
        self.profiles['res']['s_charger_ac'] = (np.array(q_chargers_res_ac)**2 + np.array(q_chargers_res_ac)**2)**0.5
        self.profiles['ind']['s_charger_ac'] = (np.array(q_chargers_ind_ac)**2 + np.array(q_chargers_ind_ac)**2)**0.5
        self.profiles['com']['s_charger_ac'] = (np.array(q_chargers_com_ac)**2 + np.array(q_chargers_com_ac)**2)**0.5

        self.profiles['res']['p_load_dc'] = np.array(p_loads_res_dc)
        self.profiles['ind']['p_load_dc'] = np.array(p_loads_ind_dc)
        self.profiles['com']['p_load_dc'] = np.array(p_loads_com_dc)
        self.profiles['res']['p_charger_dc'] = np.array(p_chargers_res_dc)
        self.profiles['ind']['p_charger_dc'] = np.array(p_chargers_ind_dc)
        self.profiles['com']['p_charger_dc'] = np.array(p_chargers_com_dc)
        
        self.times = times
        
        
def set_loads(excel_file,params,case,xy_0=False):
    
    if xy_0: 
        load_factor= 0.0 
    else: 
        load_factor = 1.0
        
    df_loads_ac = pd.read_excel(excel_file, 
                  sheet_name=case, 
                  header=1, names=None, index_col=0, 
                  usecols=[1,4,5], 
                  skiprows=[8,9,10,12,13,14,23,24,25,26])

    df_loads_dc = pd.read_excel(excel_file, 
                  sheet_name=case, 
                  header=1, names=None, index_col=0, 
                  usecols=[7,8,10], 
                  skiprows=[2,8,9,10,12,13,14,15,17,18,20,21,23,24,25,26])



    if case != 'Case 3':
        for item in df_loads_ac.index:
            p_kw   = df_loads_ac.loc[item]['P (kW)']
            q_kvar = df_loads_ac.loc[item]['Q (kvar)']
    
            for ph in ['a','b','c']:
                params.update({f'p_load_{item}_{ph}':load_factor*1e3*p_kw/3})
                params.update({f'q_load_{item}_{ph}':load_factor*1e3*q_kvar/3})
        
    if case == 'Case 3':
        for item in df_loads_ac.index:
            p_kw   = df_loads_ac.loc[item]['P (kW)']
            q_kvar = df_loads_ac.loc[item]['Q (kvar)']
    
            for ph in ['a']:
                params.update({f'p_load_{item}_{ph}':load_factor*1e3*p_kw})
                params.update({f'q_load_{item}_{ph}':load_factor*1e3*q_kvar})        
    
            for ph in ['b','c']:
                params.update({f'p_load_{item}_{ph}':0.0})
                params.update({f'q_load_{item}_{ph}':0.0})    
            
    for item in df_loads_dc.index:
        p_kw   = df_loads_dc.loc[item]['Pdc (kW)']

        for ph in ['a']:
            params.update({f'p_load_{item}_{ph}':load_factor*1e3*p_kw})
            params.update({f'q_load_{item}_{ph}':0.0})
        for ph in ['b','c']:
            params.update({f'p_load_{item}_{ph}':0.0})
            params.update({f'q_load_{item}_{ph}':0.0})

def get_head_power(grid):
    v_a_r,v_a_i,v_b_r,v_b_i,v_c_r,v_c_i = grid.get_mvalue(['v_MV0_a_r','v_MV0_a_i',
                                                           'v_MV0_b_r','v_MV0_b_i',
                                                           'v_MV0_c_r','v_MV0_c_i',])
    v_abc = np.array([v_a_r+1j*v_a_i, v_b_r+1j*v_b_i, v_c_r +1j*v_c_i])

    i_a_r,i_a_i,i_b_r,i_b_i,i_c_r,i_c_i = grid.get_mvalue(['i_t_MV0_R01_1_a_r','i_t_MV0_R01_1_a_i',
                                                           'i_t_MV0_R01_1_b_r','i_t_MV0_R01_1_b_i',
                                                           'i_t_MV0_R01_1_c_r','i_t_MV0_R01_1_c_i'])
    i_abc = np.array([i_a_r+1j*i_a_i, i_b_r+1j*i_b_i, i_c_r +1j*i_c_i])

    s_res = np.sum(v_abc*np.conj(i_abc))

    i_a_r,i_a_i,i_b_r,i_b_i,i_c_r,i_c_i = grid.get_mvalue(['i_t_MV0_I01_1_a_r','i_t_MV0_I01_1_a_i',
                                                           'i_t_MV0_I01_1_b_r','i_t_MV0_I01_1_b_i',
                                                           'i_t_MV0_I01_1_c_r','i_t_MV0_I01_1_c_i',])
    i_abc = np.array([i_a_r+1j*i_a_i, i_b_r+1j*i_b_i, i_c_r +1j*i_c_i])

    s_ind = np.sum(v_abc*np.conj(i_abc))

    i_a_r,i_a_i,i_b_r,i_b_i,i_c_r,i_c_i = grid.get_mvalue(['i_t_MV0_C01_1_a_r','i_t_MV0_C01_1_a_i',
                                                           'i_t_MV0_C01_1_b_r','i_t_MV0_C01_1_b_i',
                                                           'i_t_MV0_C01_1_c_r','i_t_MV0_C01_1_c_i',])
    i_abc = np.array([i_a_r+1j*i_a_i, i_b_r+1j*i_b_i, i_c_r +1j*i_c_i])

    s_com = np.sum(v_abc*np.conj(i_abc))
    
    s_total = s_res + s_ind + s_com
    
    return s_total

def set_v(grid,bus,V_rms,phases=['a','b','c'],phi=0.0):
        
    for ph,mult in zip(phases,[0.0,1.0,2.0]):
        v = V_rms/np.sqrt(3)*np.exp(1j*2/3*np.pi*mult)
        grid.set_value(f'v_{bus}_{ph}_r',v.real)
        grid.set_value(f'v_{bus}_{ph}_i',v.imag)
        
def get_bus_powers(gt_grid,bus_name):
    idx = gt_grid.bus_data['bus_id'].index(bus_name)
    p_a,q_a = gt_grid.buses[idx]['p_a'],gt_grid.buses[idx]['p_a']
    p_b,q_b = gt_grid.buses[idx]['p_b'],gt_grid.buses[idx]['q_b']
    p_c,q_c = gt_grid.buses[idx]['p_c'],gt_grid.buses[idx]['q_c']
    p = p_a + p_b + p_c
    q = q_a + q_b + q_c
    return p,q
    
