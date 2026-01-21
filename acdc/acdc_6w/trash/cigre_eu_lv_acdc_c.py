import numpy as np
import numba
import scipy.optimize as sopt
import scipy.sparse as sspa
from scipy.sparse.linalg import spsolve,spilu,splu
from numba import cuda
import cffi
import numba.core.typing.cffi_utils as cffi_support
from io import BytesIO
import pkgutil

dae_file_mode = 'local'

ffi = cffi.FFI()

if dae_file_mode == 'local':
    import cigre_eu_lv_acdc_c_cffi as jacs
if dae_file_mode == 'enviroment':
    import envus.no_enviroment.cigre_eu_lv_acdc_c_cffi as jacs
if dae_file_mode == 'colab':
    import cigre_eu_lv_acdc_c_cffi as jacs
    
cffi_support.register_module(jacs)
f_ini_eval = jacs.lib.f_ini_eval
g_ini_eval = jacs.lib.g_ini_eval
f_run_eval = jacs.lib.f_run_eval
g_run_eval = jacs.lib.g_run_eval
h_eval  = jacs.lib.h_eval

de_jac_ini_xy_eval = jacs.lib.de_jac_ini_xy_eval
de_jac_ini_up_eval = jacs.lib.de_jac_ini_up_eval
de_jac_ini_num_eval = jacs.lib.de_jac_ini_num_eval

sp_jac_ini_xy_eval = jacs.lib.sp_jac_ini_xy_eval
sp_jac_ini_up_eval = jacs.lib.sp_jac_ini_up_eval
sp_jac_ini_num_eval = jacs.lib.sp_jac_ini_num_eval

de_jac_run_xy_eval = jacs.lib.de_jac_run_xy_eval
de_jac_run_up_eval = jacs.lib.de_jac_run_up_eval
de_jac_run_num_eval = jacs.lib.de_jac_run_num_eval

sp_jac_run_xy_eval = jacs.lib.sp_jac_run_xy_eval
sp_jac_run_up_eval = jacs.lib.sp_jac_run_up_eval
sp_jac_run_num_eval = jacs.lib.sp_jac_run_num_eval

de_jac_trap_xy_eval= jacs.lib.de_jac_trap_xy_eval            
de_jac_trap_up_eval= jacs.lib.de_jac_trap_up_eval        
de_jac_trap_num_eval= jacs.lib.de_jac_trap_num_eval

sp_jac_trap_xy_eval= jacs.lib.sp_jac_trap_xy_eval            
sp_jac_trap_up_eval= jacs.lib.sp_jac_trap_up_eval        
sp_jac_trap_num_eval= jacs.lib.sp_jac_trap_num_eval

sp_Fu_run_up_eval = jacs.lib.sp_Fu_run_up_eval
sp_Gu_run_up_eval = jacs.lib.sp_Gu_run_up_eval
sp_Hx_run_up_eval = jacs.lib.sp_Hx_run_up_eval
sp_Hy_run_up_eval = jacs.lib.sp_Hy_run_up_eval
sp_Hu_run_up_eval = jacs.lib.sp_Hu_run_up_eval
sp_Fu_run_xy_eval = jacs.lib.sp_Fu_run_xy_eval
sp_Gu_run_xy_eval = jacs.lib.sp_Gu_run_xy_eval
sp_Hx_run_xy_eval = jacs.lib.sp_Hx_run_xy_eval
sp_Hy_run_xy_eval = jacs.lib.sp_Hy_run_xy_eval
sp_Hu_run_xy_eval = jacs.lib.sp_Hu_run_xy_eval



import json

sin = np.sin
cos = np.cos
atan2 = np.arctan2
sqrt = np.sqrt 
sign = np.sign 
exp = np.exp


class model: 

    def __init__(self): 
        
        self.matrices_folder = 'build'
        
        self.dae_file_mode = 'local'
        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 25
        self.N_y = 672 
        self.N_z = 507 
        self.N_store = 100000 
        self.params_list = ['g_shunt_R01_3', 'b_shunt_R01_3', 'g_shunt_R02_3', 'b_shunt_R02_3', 'g_shunt_R04_3', 'b_shunt_R04_3', 'g_shunt_R06_3', 'b_shunt_R06_3', 'g_shunt_R08_3', 'b_shunt_R08_3', 'g_shunt_R10_3', 'b_shunt_R10_3', 'g_shunt_R13_3', 'b_shunt_R13_3', 'g_shunt_R11_3', 'b_shunt_R11_3', 'g_shunt_R15_3', 'b_shunt_R15_3', 'g_shunt_R16_3', 'b_shunt_R16_3', 'g_shunt_R17_3', 'b_shunt_R17_3', 'g_shunt_R18_3', 'b_shunt_R18_3', 'g_shunt_I01_3', 'b_shunt_I01_3', 'g_shunt_I02_3', 'b_shunt_I02_3', 'g_shunt_C01_3', 'b_shunt_C01_3', 'g_shunt_C03_3', 'b_shunt_C03_3', 'g_shunt_C05_3', 'b_shunt_C05_3', 'g_shunt_C07_3', 'b_shunt_C07_3', 'g_shunt_C09_3', 'b_shunt_C09_3', 'g_shunt_C11_3', 'b_shunt_C11_3', 'g_shunt_C12_3', 'b_shunt_C12_3', 'g_shunt_C13_3', 'b_shunt_C13_3', 'g_shunt_C14_3', 'b_shunt_C14_3', 'g_shunt_C16_3', 'b_shunt_C16_3', 'g_shunt_C17_3', 'b_shunt_C17_3', 'g_shunt_C18_3', 'b_shunt_C18_3', 'g_shunt_C19_3', 'b_shunt_C19_3', 'g_shunt_C20_3', 'b_shunt_C20_3', 'K_abc_R01', 'K_abc_R11', 'K_abc_R15', 'K_abc_R16', 'K_abc_R17', 'K_abc_R18', 'K_abc_I02', 'K_abc_C01', 'K_abc_C12', 'K_abc_C13', 'K_abc_C14', 'K_abc_C17', 'K_abc_C18', 'K_abc_C19', 'K_abc_C20', 'A_loss_I01', 'B_loss_I01', 'C_loss_I01', 'C_a_I01', 'C_b_I01', 'C_c_I01', 'R_dc_H01', 'K_dc_H01', 'R_gdc_H01', 'A_loss_R01', 'B_loss_R01', 'C_loss_R01', 'R_dc_S01', 'K_dc_S01', 'R_gdc_S01', 'A_loss_C01', 'B_loss_C01', 'C_loss_C01', 'R_dc_D01', 'K_dc_D01', 'R_gdc_D01', 'A_loss_R10', 'B_loss_R10', 'C_loss_R10', 'R_dc_S10', 'K_dc_S10', 'R_gdc_S10', 'A_loss_R14', 'B_loss_R14', 'C_loss_R14', 'R_dc_S14', 'K_dc_S14', 'R_gdc_S14', 'A_loss_I02', 'B_loss_I02', 'C_loss_I02', 'R_dc_H02', 'K_dc_H02', 'R_gdc_H02', 'A_loss_C09', 'B_loss_C09', 'C_loss_C09', 'R_dc_D09', 'K_dc_D09', 'R_gdc_D09', 'A_loss_C11', 'B_loss_C11', 'C_loss_C11', 'R_dc_D11', 'K_dc_D11', 'R_gdc_D11', 'A_loss_C16', 'B_loss_C16', 'C_loss_C16', 'R_dc_D16', 'K_dc_D16', 'R_gdc_D16', 'X_MV0_s', 'R_MV0_s', 'X_MV0_sn', 'R_MV0_sn', 'X_MV0_ng', 'R_MV0_ng', 'K_p_agc', 'K_i_agc', 'K_xif', 'K_droop_R01', 'T_droop_R01', 'K_droop_R10', 'T_droop_R10', 'K_droop_R14', 'T_droop_R14', 'K_droop_I02', 'T_droop_I02', 'K_droop_C01', 'T_droop_C01', 'K_droop_C09', 'T_droop_C09', 'K_droop_C11', 'T_droop_C11', 'K_droop_C16', 'T_droop_C16'] 
        self.params_values_list  = [0.3333333333333333, 0.0, 0.025, 0.0, 0.025, 0.0, 0.025, 0.0, 0.025, 0.0, 0.025, 0.0, 0.025, 0.0, 0.025, 0.0, 0.025, 0.0, 0.025, 0.0, 0.025, 0.0, 0.025, 0.0, 0.3333333333333333, 0.0, 0.025, 0.0, 0.3333333333333333, 0.0, 0.025, 0.0, 0.025, 0.0, 0.025, 0.0, 0.025, 0.0, 0.025, 0.0, 0.025, 0.0, 0.025, 0.0, 0.025, 0.0, 0.025, 0.0, 0.025, 0.0, 0.025, 0.0, 0.025, 0.0, 0.025, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.92, 0.45, 0.027, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 1e-06, 1e-06, 3.0, 2.92, 0.45, 0.027, 1e-06, 1e-06, 3.0, 2.92, 0.45, 0.027, 1e-06, 1e-06, 3.0, 2.92, 0.45, 0.027, 1e-06, 1e-06, 3.0, 2.92, 0.45, 0.027, 1e-06, 1e-06, 3.0, 2.92, 0.45, 0.027, 1e-06, 1e-06, 3.0, 2.92, 0.45, 0.027, 1e-06, 1e-06, 3.0, 2.92, 0.45, 0.027, 1e-06, 1e-06, 3.0, 2.92, 0.45, 0.027, 1e-06, 1e-06, 3.0, 0.1, 0.01, 0.1, 0.01, 0.0, 3.0, 0.01, 0.01, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0] 
        self.inputs_ini_list = ['p_load_R01_a', 'q_load_R01_a', 'g_load_R01_a', 'b_load_R01_a', 'p_load_R01_b', 'q_load_R01_b', 'g_load_R01_b', 'b_load_R01_b', 'p_load_R01_c', 'q_load_R01_c', 'g_load_R01_c', 'b_load_R01_c', 'p_load_R11_a', 'q_load_R11_a', 'g_load_R11_a', 'b_load_R11_a', 'p_load_R11_b', 'q_load_R11_b', 'g_load_R11_b', 'b_load_R11_b', 'p_load_R11_c', 'q_load_R11_c', 'g_load_R11_c', 'b_load_R11_c', 'p_load_R15_a', 'q_load_R15_a', 'g_load_R15_a', 'b_load_R15_a', 'p_load_R15_b', 'q_load_R15_b', 'g_load_R15_b', 'b_load_R15_b', 'p_load_R15_c', 'q_load_R15_c', 'g_load_R15_c', 'b_load_R15_c', 'p_load_R16_a', 'q_load_R16_a', 'g_load_R16_a', 'b_load_R16_a', 'p_load_R16_b', 'q_load_R16_b', 'g_load_R16_b', 'b_load_R16_b', 'p_load_R16_c', 'q_load_R16_c', 'g_load_R16_c', 'b_load_R16_c', 'p_load_R17_a', 'q_load_R17_a', 'g_load_R17_a', 'b_load_R17_a', 'p_load_R17_b', 'q_load_R17_b', 'g_load_R17_b', 'b_load_R17_b', 'p_load_R17_c', 'q_load_R17_c', 'g_load_R17_c', 'b_load_R17_c', 'p_load_R18_a', 'q_load_R18_a', 'g_load_R18_a', 'b_load_R18_a', 'p_load_R18_b', 'q_load_R18_b', 'g_load_R18_b', 'b_load_R18_b', 'p_load_R18_c', 'q_load_R18_c', 'g_load_R18_c', 'b_load_R18_c', 'p_load_I02_a', 'q_load_I02_a', 'g_load_I02_a', 'b_load_I02_a', 'p_load_I02_b', 'q_load_I02_b', 'g_load_I02_b', 'b_load_I02_b', 'p_load_I02_c', 'q_load_I02_c', 'g_load_I02_c', 'b_load_I02_c', 'p_load_C01_a', 'q_load_C01_a', 'g_load_C01_a', 'b_load_C01_a', 'p_load_C01_b', 'q_load_C01_b', 'g_load_C01_b', 'b_load_C01_b', 'p_load_C01_c', 'q_load_C01_c', 'g_load_C01_c', 'b_load_C01_c', 'p_load_C12_a', 'q_load_C12_a', 'g_load_C12_a', 'b_load_C12_a', 'p_load_C12_b', 'q_load_C12_b', 'g_load_C12_b', 'b_load_C12_b', 'p_load_C12_c', 'q_load_C12_c', 'g_load_C12_c', 'b_load_C12_c', 'p_load_C13_a', 'q_load_C13_a', 'g_load_C13_a', 'b_load_C13_a', 'p_load_C13_b', 'q_load_C13_b', 'g_load_C13_b', 'b_load_C13_b', 'p_load_C13_c', 'q_load_C13_c', 'g_load_C13_c', 'b_load_C13_c', 'p_load_C14_a', 'q_load_C14_a', 'g_load_C14_a', 'b_load_C14_a', 'p_load_C14_b', 'q_load_C14_b', 'g_load_C14_b', 'b_load_C14_b', 'p_load_C14_c', 'q_load_C14_c', 'g_load_C14_c', 'b_load_C14_c', 'p_load_C17_a', 'q_load_C17_a', 'g_load_C17_a', 'b_load_C17_a', 'p_load_C17_b', 'q_load_C17_b', 'g_load_C17_b', 'b_load_C17_b', 'p_load_C17_c', 'q_load_C17_c', 'g_load_C17_c', 'b_load_C17_c', 'p_load_C18_a', 'q_load_C18_a', 'g_load_C18_a', 'b_load_C18_a', 'p_load_C18_b', 'q_load_C18_b', 'g_load_C18_b', 'b_load_C18_b', 'p_load_C18_c', 'q_load_C18_c', 'g_load_C18_c', 'b_load_C18_c', 'p_load_C19_a', 'q_load_C19_a', 'g_load_C19_a', 'b_load_C19_a', 'p_load_C19_b', 'q_load_C19_b', 'g_load_C19_b', 'b_load_C19_b', 'p_load_C19_c', 'q_load_C19_c', 'g_load_C19_c', 'b_load_C19_c', 'p_load_C20_a', 'q_load_C20_a', 'g_load_C20_a', 'b_load_C20_a', 'p_load_C20_b', 'q_load_C20_b', 'g_load_C20_b', 'b_load_C20_b', 'p_load_C20_c', 'q_load_C20_c', 'g_load_C20_c', 'b_load_C20_c', 'p_load_S15', 'p_load_S11', 'p_load_S16', 'p_load_S17', 'p_load_S18', 'p_load_H02', 'p_load_D12', 'p_load_D17', 'p_load_D19', 'p_load_D20', 'v_dc_H01_ref', 'q_vsc_a_I01', 'q_vsc_b_I01', 'q_vsc_c_I01', 'v_dc_S01_ref', 'q_vsc_a_R01', 'q_vsc_b_R01', 'q_vsc_c_R01', 'v_dc_D01_ref', 'q_vsc_a_C01', 'q_vsc_b_C01', 'q_vsc_c_C01', 'v_dc_S10_ref', 'q_vsc_a_R10', 'q_vsc_b_R10', 'q_vsc_c_R10', 'v_dc_S14_ref', 'q_vsc_a_R14', 'q_vsc_b_R14', 'q_vsc_c_R14', 'v_dc_H02_ref', 'q_vsc_a_I02', 'q_vsc_b_I02', 'q_vsc_c_I02', 'v_dc_D09_ref', 'q_vsc_a_C09', 'q_vsc_b_C09', 'q_vsc_c_C09', 'v_dc_D11_ref', 'q_vsc_a_C11', 'q_vsc_b_C11', 'q_vsc_c_C11', 'v_dc_D16_ref', 'q_vsc_a_C16', 'q_vsc_b_C16', 'q_vsc_c_C16', 'e_ao_m_MV0', 'e_bo_m_MV0', 'e_co_m_MV0', 'phi_MV0', 'u_freq'] 
        self.inputs_ini_values_list  = [1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 30000.0, 30000.0, 30000.0, 30000.0, 30000.0, 30000.0, 30000.0, 30000.0, 0.0, 30000.0, 800.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 11547.005383792515, 11547.005383792515, 11547.005383792515, 0.0, 0.0] 
        self.inputs_run_list = ['p_load_R01_a', 'q_load_R01_a', 'g_load_R01_a', 'b_load_R01_a', 'p_load_R01_b', 'q_load_R01_b', 'g_load_R01_b', 'b_load_R01_b', 'p_load_R01_c', 'q_load_R01_c', 'g_load_R01_c', 'b_load_R01_c', 'p_load_R11_a', 'q_load_R11_a', 'g_load_R11_a', 'b_load_R11_a', 'p_load_R11_b', 'q_load_R11_b', 'g_load_R11_b', 'b_load_R11_b', 'p_load_R11_c', 'q_load_R11_c', 'g_load_R11_c', 'b_load_R11_c', 'p_load_R15_a', 'q_load_R15_a', 'g_load_R15_a', 'b_load_R15_a', 'p_load_R15_b', 'q_load_R15_b', 'g_load_R15_b', 'b_load_R15_b', 'p_load_R15_c', 'q_load_R15_c', 'g_load_R15_c', 'b_load_R15_c', 'p_load_R16_a', 'q_load_R16_a', 'g_load_R16_a', 'b_load_R16_a', 'p_load_R16_b', 'q_load_R16_b', 'g_load_R16_b', 'b_load_R16_b', 'p_load_R16_c', 'q_load_R16_c', 'g_load_R16_c', 'b_load_R16_c', 'p_load_R17_a', 'q_load_R17_a', 'g_load_R17_a', 'b_load_R17_a', 'p_load_R17_b', 'q_load_R17_b', 'g_load_R17_b', 'b_load_R17_b', 'p_load_R17_c', 'q_load_R17_c', 'g_load_R17_c', 'b_load_R17_c', 'p_load_R18_a', 'q_load_R18_a', 'g_load_R18_a', 'b_load_R18_a', 'p_load_R18_b', 'q_load_R18_b', 'g_load_R18_b', 'b_load_R18_b', 'p_load_R18_c', 'q_load_R18_c', 'g_load_R18_c', 'b_load_R18_c', 'p_load_I02_a', 'q_load_I02_a', 'g_load_I02_a', 'b_load_I02_a', 'p_load_I02_b', 'q_load_I02_b', 'g_load_I02_b', 'b_load_I02_b', 'p_load_I02_c', 'q_load_I02_c', 'g_load_I02_c', 'b_load_I02_c', 'p_load_C01_a', 'q_load_C01_a', 'g_load_C01_a', 'b_load_C01_a', 'p_load_C01_b', 'q_load_C01_b', 'g_load_C01_b', 'b_load_C01_b', 'p_load_C01_c', 'q_load_C01_c', 'g_load_C01_c', 'b_load_C01_c', 'p_load_C12_a', 'q_load_C12_a', 'g_load_C12_a', 'b_load_C12_a', 'p_load_C12_b', 'q_load_C12_b', 'g_load_C12_b', 'b_load_C12_b', 'p_load_C12_c', 'q_load_C12_c', 'g_load_C12_c', 'b_load_C12_c', 'p_load_C13_a', 'q_load_C13_a', 'g_load_C13_a', 'b_load_C13_a', 'p_load_C13_b', 'q_load_C13_b', 'g_load_C13_b', 'b_load_C13_b', 'p_load_C13_c', 'q_load_C13_c', 'g_load_C13_c', 'b_load_C13_c', 'p_load_C14_a', 'q_load_C14_a', 'g_load_C14_a', 'b_load_C14_a', 'p_load_C14_b', 'q_load_C14_b', 'g_load_C14_b', 'b_load_C14_b', 'p_load_C14_c', 'q_load_C14_c', 'g_load_C14_c', 'b_load_C14_c', 'p_load_C17_a', 'q_load_C17_a', 'g_load_C17_a', 'b_load_C17_a', 'p_load_C17_b', 'q_load_C17_b', 'g_load_C17_b', 'b_load_C17_b', 'p_load_C17_c', 'q_load_C17_c', 'g_load_C17_c', 'b_load_C17_c', 'p_load_C18_a', 'q_load_C18_a', 'g_load_C18_a', 'b_load_C18_a', 'p_load_C18_b', 'q_load_C18_b', 'g_load_C18_b', 'b_load_C18_b', 'p_load_C18_c', 'q_load_C18_c', 'g_load_C18_c', 'b_load_C18_c', 'p_load_C19_a', 'q_load_C19_a', 'g_load_C19_a', 'b_load_C19_a', 'p_load_C19_b', 'q_load_C19_b', 'g_load_C19_b', 'b_load_C19_b', 'p_load_C19_c', 'q_load_C19_c', 'g_load_C19_c', 'b_load_C19_c', 'p_load_C20_a', 'q_load_C20_a', 'g_load_C20_a', 'b_load_C20_a', 'p_load_C20_b', 'q_load_C20_b', 'g_load_C20_b', 'b_load_C20_b', 'p_load_C20_c', 'q_load_C20_c', 'g_load_C20_c', 'b_load_C20_c', 'p_load_S15', 'p_load_S11', 'p_load_S16', 'p_load_S17', 'p_load_S18', 'p_load_H02', 'p_load_D12', 'p_load_D17', 'p_load_D19', 'p_load_D20', 'v_dc_H01_ref', 'q_vsc_a_I01', 'q_vsc_b_I01', 'q_vsc_c_I01', 'v_dc_S01_ref', 'q_vsc_a_R01', 'q_vsc_b_R01', 'q_vsc_c_R01', 'v_dc_D01_ref', 'q_vsc_a_C01', 'q_vsc_b_C01', 'q_vsc_c_C01', 'v_dc_S10_ref', 'q_vsc_a_R10', 'q_vsc_b_R10', 'q_vsc_c_R10', 'v_dc_S14_ref', 'q_vsc_a_R14', 'q_vsc_b_R14', 'q_vsc_c_R14', 'v_dc_H02_ref', 'q_vsc_a_I02', 'q_vsc_b_I02', 'q_vsc_c_I02', 'v_dc_D09_ref', 'q_vsc_a_C09', 'q_vsc_b_C09', 'q_vsc_c_C09', 'v_dc_D11_ref', 'q_vsc_a_C11', 'q_vsc_b_C11', 'q_vsc_c_C11', 'v_dc_D16_ref', 'q_vsc_a_C16', 'q_vsc_b_C16', 'q_vsc_c_C16', 'e_ao_m_MV0', 'e_bo_m_MV0', 'e_co_m_MV0', 'phi_MV0', 'u_freq'] 
        self.inputs_run_values_list = [1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 1000.0, 0, 0, 0, 30000.0, 30000.0, 30000.0, 30000.0, 30000.0, 30000.0, 30000.0, 30000.0, 0.0, 30000.0, 800.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 11547.005383792515, 11547.005383792515, 11547.005383792515, 0.0, 0.0] 
        self.outputs_list = ['i_l_R01_0_R02_0_r', 'i_l_R01_0_R02_0_i', 'i_l_R01_1_R02_1_r', 'i_l_R01_1_R02_1_i', 'i_l_R01_2_R02_2_r', 'i_l_R01_2_R02_2_i', 'i_l_R01_3_R02_3_r', 'i_l_R01_3_R02_3_i', 'i_l_R02_0_R03_0_r', 'i_l_R02_0_R03_0_i', 'i_l_R02_1_R03_1_r', 'i_l_R02_1_R03_1_i', 'i_l_R02_2_R03_2_r', 'i_l_R02_2_R03_2_i', 'i_l_R02_3_R03_3_r', 'i_l_R02_3_R03_3_i', 'i_l_R03_0_R04_0_r', 'i_l_R03_0_R04_0_i', 'i_l_R03_1_R04_1_r', 'i_l_R03_1_R04_1_i', 'i_l_R03_2_R04_2_r', 'i_l_R03_2_R04_2_i', 'i_l_R03_3_R04_3_r', 'i_l_R03_3_R04_3_i', 'i_l_R04_0_R05_0_r', 'i_l_R04_0_R05_0_i', 'i_l_R04_1_R05_1_r', 'i_l_R04_1_R05_1_i', 'i_l_R04_2_R05_2_r', 'i_l_R04_2_R05_2_i', 'i_l_R04_3_R05_3_r', 'i_l_R04_3_R05_3_i', 'i_l_R05_0_R06_0_r', 'i_l_R05_0_R06_0_i', 'i_l_R05_1_R06_1_r', 'i_l_R05_1_R06_1_i', 'i_l_R05_2_R06_2_r', 'i_l_R05_2_R06_2_i', 'i_l_R05_3_R06_3_r', 'i_l_R05_3_R06_3_i', 'i_l_R06_0_R07_0_r', 'i_l_R06_0_R07_0_i', 'i_l_R06_1_R07_1_r', 'i_l_R06_1_R07_1_i', 'i_l_R06_2_R07_2_r', 'i_l_R06_2_R07_2_i', 'i_l_R06_3_R07_3_r', 'i_l_R06_3_R07_3_i', 'i_l_R07_0_R08_0_r', 'i_l_R07_0_R08_0_i', 'i_l_R07_1_R08_1_r', 'i_l_R07_1_R08_1_i', 'i_l_R07_2_R08_2_r', 'i_l_R07_2_R08_2_i', 'i_l_R07_3_R08_3_r', 'i_l_R07_3_R08_3_i', 'i_l_R08_0_R09_0_r', 'i_l_R08_0_R09_0_i', 'i_l_R08_1_R09_1_r', 'i_l_R08_1_R09_1_i', 'i_l_R08_2_R09_2_r', 'i_l_R08_2_R09_2_i', 'i_l_R08_3_R09_3_r', 'i_l_R08_3_R09_3_i', 'i_l_R09_0_R10_0_r', 'i_l_R09_0_R10_0_i', 'i_l_R09_1_R10_1_r', 'i_l_R09_1_R10_1_i', 'i_l_R09_2_R10_2_r', 'i_l_R09_2_R10_2_i', 'i_l_R09_3_R10_3_r', 'i_l_R09_3_R10_3_i', 'i_l_R03_0_R11_0_r', 'i_l_R03_0_R11_0_i', 'i_l_R03_1_R11_1_r', 'i_l_R03_1_R11_1_i', 'i_l_R03_2_R11_2_r', 'i_l_R03_2_R11_2_i', 'i_l_R03_3_R11_3_r', 'i_l_R03_3_R11_3_i', 'i_l_R04_0_R12_0_r', 'i_l_R04_0_R12_0_i', 'i_l_R04_1_R12_1_r', 'i_l_R04_1_R12_1_i', 'i_l_R04_2_R12_2_r', 'i_l_R04_2_R12_2_i', 'i_l_R04_3_R12_3_r', 'i_l_R04_3_R12_3_i', 'i_l_R12_0_R13_0_r', 'i_l_R12_0_R13_0_i', 'i_l_R12_1_R13_1_r', 'i_l_R12_1_R13_1_i', 'i_l_R12_2_R13_2_r', 'i_l_R12_2_R13_2_i', 'i_l_R12_3_R13_3_r', 'i_l_R12_3_R13_3_i', 'i_l_R13_0_R14_0_r', 'i_l_R13_0_R14_0_i', 'i_l_R13_1_R14_1_r', 'i_l_R13_1_R14_1_i', 'i_l_R13_2_R14_2_r', 'i_l_R13_2_R14_2_i', 'i_l_R13_3_R14_3_r', 'i_l_R13_3_R14_3_i', 'i_l_R14_0_R15_0_r', 'i_l_R14_0_R15_0_i', 'i_l_R14_1_R15_1_r', 'i_l_R14_1_R15_1_i', 'i_l_R14_2_R15_2_r', 'i_l_R14_2_R15_2_i', 'i_l_R14_3_R15_3_r', 'i_l_R14_3_R15_3_i', 'i_l_R06_0_R16_0_r', 'i_l_R06_0_R16_0_i', 'i_l_R06_1_R16_1_r', 'i_l_R06_1_R16_1_i', 'i_l_R06_2_R16_2_r', 'i_l_R06_2_R16_2_i', 'i_l_R06_3_R16_3_r', 'i_l_R06_3_R16_3_i', 'i_l_R09_0_R17_0_r', 'i_l_R09_0_R17_0_i', 'i_l_R09_1_R17_1_r', 'i_l_R09_1_R17_1_i', 'i_l_R09_2_R17_2_r', 'i_l_R09_2_R17_2_i', 'i_l_R09_3_R17_3_r', 'i_l_R09_3_R17_3_i', 'i_l_R10_0_R18_0_r', 'i_l_R10_0_R18_0_i', 'i_l_R10_1_R18_1_r', 'i_l_R10_1_R18_1_i', 'i_l_R10_2_R18_2_r', 'i_l_R10_2_R18_2_i', 'i_l_R10_3_R18_3_r', 'i_l_R10_3_R18_3_i', 'i_l_I01_0_I02_0_r', 'i_l_I01_0_I02_0_i', 'i_l_I01_1_I02_1_r', 'i_l_I01_1_I02_1_i', 'i_l_I01_2_I02_2_r', 'i_l_I01_2_I02_2_i', 'i_l_I01_3_I02_3_r', 'i_l_I01_3_I02_3_i', 'i_l_C01_0_C02_0_r', 'i_l_C01_0_C02_0_i', 'i_l_C01_1_C02_1_r', 'i_l_C01_1_C02_1_i', 'i_l_C01_2_C02_2_r', 'i_l_C01_2_C02_2_i', 'i_l_C01_3_C02_3_r', 'i_l_C01_3_C02_3_i', 'i_l_C02_0_C03_0_r', 'i_l_C02_0_C03_0_i', 'i_l_C02_1_C03_1_r', 'i_l_C02_1_C03_1_i', 'i_l_C02_2_C03_2_r', 'i_l_C02_2_C03_2_i', 'i_l_C02_3_C03_3_r', 'i_l_C02_3_C03_3_i', 'i_l_C03_0_C04_0_r', 'i_l_C03_0_C04_0_i', 'i_l_C03_1_C04_1_r', 'i_l_C03_1_C04_1_i', 'i_l_C03_2_C04_2_r', 'i_l_C03_2_C04_2_i', 'i_l_C03_3_C04_3_r', 'i_l_C03_3_C04_3_i', 'i_l_C04_0_C05_0_r', 'i_l_C04_0_C05_0_i', 'i_l_C04_1_C05_1_r', 'i_l_C04_1_C05_1_i', 'i_l_C04_2_C05_2_r', 'i_l_C04_2_C05_2_i', 'i_l_C04_3_C05_3_r', 'i_l_C04_3_C05_3_i', 'i_l_C05_0_C06_0_r', 'i_l_C05_0_C06_0_i', 'i_l_C05_1_C06_1_r', 'i_l_C05_1_C06_1_i', 'i_l_C05_2_C06_2_r', 'i_l_C05_2_C06_2_i', 'i_l_C05_3_C06_3_r', 'i_l_C05_3_C06_3_i', 'i_l_C06_0_C07_0_r', 'i_l_C06_0_C07_0_i', 'i_l_C06_1_C07_1_r', 'i_l_C06_1_C07_1_i', 'i_l_C06_2_C07_2_r', 'i_l_C06_2_C07_2_i', 'i_l_C06_3_C07_3_r', 'i_l_C06_3_C07_3_i', 'i_l_C07_0_C08_0_r', 'i_l_C07_0_C08_0_i', 'i_l_C07_1_C08_1_r', 'i_l_C07_1_C08_1_i', 'i_l_C07_2_C08_2_r', 'i_l_C07_2_C08_2_i', 'i_l_C07_3_C08_3_r', 'i_l_C07_3_C08_3_i', 'i_l_C08_0_C09_0_r', 'i_l_C08_0_C09_0_i', 'i_l_C08_1_C09_1_r', 'i_l_C08_1_C09_1_i', 'i_l_C08_2_C09_2_r', 'i_l_C08_2_C09_2_i', 'i_l_C08_3_C09_3_r', 'i_l_C08_3_C09_3_i', 'i_l_C03_0_C10_0_r', 'i_l_C03_0_C10_0_i', 'i_l_C03_1_C10_1_r', 'i_l_C03_1_C10_1_i', 'i_l_C03_2_C10_2_r', 'i_l_C03_2_C10_2_i', 'i_l_C03_3_C10_3_r', 'i_l_C03_3_C10_3_i', 'i_l_C10_0_C11_0_r', 'i_l_C10_0_C11_0_i', 'i_l_C10_1_C11_1_r', 'i_l_C10_1_C11_1_i', 'i_l_C10_2_C11_2_r', 'i_l_C10_2_C11_2_i', 'i_l_C10_3_C11_3_r', 'i_l_C10_3_C11_3_i', 'i_l_C11_0_C12_0_r', 'i_l_C11_0_C12_0_i', 'i_l_C11_1_C12_1_r', 'i_l_C11_1_C12_1_i', 'i_l_C11_2_C12_2_r', 'i_l_C11_2_C12_2_i', 'i_l_C11_3_C12_3_r', 'i_l_C11_3_C12_3_i', 'i_l_C11_0_C13_0_r', 'i_l_C11_0_C13_0_i', 'i_l_C11_1_C13_1_r', 'i_l_C11_1_C13_1_i', 'i_l_C11_2_C13_2_r', 'i_l_C11_2_C13_2_i', 'i_l_C11_3_C13_3_r', 'i_l_C11_3_C13_3_i', 'i_l_C10_0_C14_0_r', 'i_l_C10_0_C14_0_i', 'i_l_C10_1_C14_1_r', 'i_l_C10_1_C14_1_i', 'i_l_C10_2_C14_2_r', 'i_l_C10_2_C14_2_i', 'i_l_C10_3_C14_3_r', 'i_l_C10_3_C14_3_i', 'i_l_C05_0_C15_0_r', 'i_l_C05_0_C15_0_i', 'i_l_C05_1_C15_1_r', 'i_l_C05_1_C15_1_i', 'i_l_C05_2_C15_2_r', 'i_l_C05_2_C15_2_i', 'i_l_C05_3_C15_3_r', 'i_l_C05_3_C15_3_i', 'i_l_C15_0_C16_0_r', 'i_l_C15_0_C16_0_i', 'i_l_C15_1_C16_1_r', 'i_l_C15_1_C16_1_i', 'i_l_C15_2_C16_2_r', 'i_l_C15_2_C16_2_i', 'i_l_C15_3_C16_3_r', 'i_l_C15_3_C16_3_i', 'i_l_C15_0_C18_0_r', 'i_l_C15_0_C18_0_i', 'i_l_C15_1_C18_1_r', 'i_l_C15_1_C18_1_i', 'i_l_C15_2_C18_2_r', 'i_l_C15_2_C18_2_i', 'i_l_C15_3_C18_3_r', 'i_l_C15_3_C18_3_i', 'i_l_C16_0_C17_0_r', 'i_l_C16_0_C17_0_i', 'i_l_C16_1_C17_1_r', 'i_l_C16_1_C17_1_i', 'i_l_C16_2_C17_2_r', 'i_l_C16_2_C17_2_i', 'i_l_C16_3_C17_3_r', 'i_l_C16_3_C17_3_i', 'i_l_C08_0_C19_0_r', 'i_l_C08_0_C19_0_i', 'i_l_C08_1_C19_1_r', 'i_l_C08_1_C19_1_i', 'i_l_C08_2_C19_2_r', 'i_l_C08_2_C19_2_i', 'i_l_C08_3_C19_3_r', 'i_l_C08_3_C19_3_i', 'i_l_C09_0_C20_0_r', 'i_l_C09_0_C20_0_i', 'i_l_C09_1_C20_1_r', 'i_l_C09_1_C20_1_i', 'i_l_C09_2_C20_2_r', 'i_l_C09_2_C20_2_i', 'i_l_C09_3_C20_3_r', 'i_l_C09_3_C20_3_i', 'i_l_S01_0_S03_0_r', 'i_l_S01_0_S03_0_i', 'i_l_S01_1_S03_1_r', 'i_l_S01_1_S03_1_i', 'i_l_S03_0_S04_0_r', 'i_l_S03_0_S04_0_i', 'i_l_S03_1_S04_1_r', 'i_l_S03_1_S04_1_i', 'i_l_S04_0_S06_0_r', 'i_l_S04_0_S06_0_i', 'i_l_S04_1_S06_1_r', 'i_l_S04_1_S06_1_i', 'i_l_S06_0_S07_0_r', 'i_l_S06_0_S07_0_i', 'i_l_S06_1_S07_1_r', 'i_l_S06_1_S07_1_i', 'i_l_S07_0_S09_0_r', 'i_l_S07_0_S09_0_i', 'i_l_S07_1_S09_1_r', 'i_l_S07_1_S09_1_i', 'i_l_S09_0_S10_0_r', 'i_l_S09_0_S10_0_i', 'i_l_S09_1_S10_1_r', 'i_l_S09_1_S10_1_i', 'i_l_S03_0_S11_0_r', 'i_l_S03_0_S11_0_i', 'i_l_S03_1_S11_1_r', 'i_l_S03_1_S11_1_i', 'i_l_S04_0_S14_0_r', 'i_l_S04_0_S14_0_i', 'i_l_S04_1_S14_1_r', 'i_l_S04_1_S14_1_i', 'i_l_S14_0_S15_0_r', 'i_l_S14_0_S15_0_i', 'i_l_S14_1_S15_1_r', 'i_l_S14_1_S15_1_i', 'i_l_S06_0_S16_0_r', 'i_l_S06_0_S16_0_i', 'i_l_S06_1_S16_1_r', 'i_l_S06_1_S16_1_i', 'i_l_S09_0_S17_0_r', 'i_l_S09_0_S17_0_i', 'i_l_S09_1_S17_1_r', 'i_l_S09_1_S17_1_i', 'i_l_S10_0_S18_0_r', 'i_l_S10_0_S18_0_i', 'i_l_S10_1_S18_1_r', 'i_l_S10_1_S18_1_i', 'i_l_H01_0_H02_0_r', 'i_l_H01_0_H02_0_i', 'i_l_H01_1_H02_1_r', 'i_l_H01_1_H02_1_i', 'i_l_D01_0_D03_0_r', 'i_l_D01_0_D03_0_i', 'i_l_D01_1_D03_1_r', 'i_l_D01_1_D03_1_i', 'i_l_D03_0_D05_0_r', 'i_l_D03_0_D05_0_i', 'i_l_D03_1_D05_1_r', 'i_l_D03_1_D05_1_i', 'i_l_D05_0_D08_0_r', 'i_l_D05_0_D08_0_i', 'i_l_D05_1_D08_1_r', 'i_l_D05_1_D08_1_i', 'i_l_D08_0_D09_0_r', 'i_l_D08_0_D09_0_i', 'i_l_D08_1_D09_1_r', 'i_l_D08_1_D09_1_i', 'i_l_D03_0_D11_0_r', 'i_l_D03_0_D11_0_i', 'i_l_D03_1_D11_1_r', 'i_l_D03_1_D11_1_i', 'i_l_D11_0_D12_0_r', 'i_l_D11_0_D12_0_i', 'i_l_D11_1_D12_1_r', 'i_l_D11_1_D12_1_i', 'i_l_D05_0_D16_0_r', 'i_l_D05_0_D16_0_i', 'i_l_D05_1_D16_1_r', 'i_l_D05_1_D16_1_i', 'i_l_D16_0_D17_0_r', 'i_l_D16_0_D17_0_i', 'i_l_D16_1_D17_1_r', 'i_l_D16_1_D17_1_i', 'i_l_D08_0_D19_0_r', 'i_l_D08_0_D19_0_i', 'i_l_D08_1_D19_1_r', 'i_l_D08_1_D19_1_i', 'i_l_D09_0_D20_0_r', 'i_l_D09_0_D20_0_i', 'i_l_D09_1_D20_1_r', 'i_l_D09_1_D20_1_i', 'i_l_S07_0_H02_0_r', 'i_l_S07_0_H02_0_i', 'i_l_S07_1_H02_1_r', 'i_l_S07_1_H02_1_i', 'i_l_H02_0_D19_0_r', 'i_l_H02_0_D19_0_i', 'i_l_H02_1_D19_1_r', 'i_l_H02_1_D19_1_i', 'i_t_MV0_R01_1_0_r', 'i_t_MV0_R01_1_0_i', 'i_t_MV0_R01_1_1_r', 'i_t_MV0_R01_1_1_i', 'i_t_MV0_R01_1_2_r', 'i_t_MV0_R01_1_2_i', 'i_t_MV0_R01_2_0_r', 'i_t_MV0_R01_2_0_i', 'i_t_MV0_R01_2_1_r', 'i_t_MV0_R01_2_1_i', 'i_t_MV0_R01_2_2_r', 'i_t_MV0_R01_2_2_i', 'i_t_MV0_R01_2_3_r', 'i_t_MV0_R01_2_3_i', 'i_t_MV0_I01_1_0_r', 'i_t_MV0_I01_1_0_i', 'i_t_MV0_I01_1_1_r', 'i_t_MV0_I01_1_1_i', 'i_t_MV0_I01_1_2_r', 'i_t_MV0_I01_1_2_i', 'i_t_MV0_I01_2_0_r', 'i_t_MV0_I01_2_0_i', 'i_t_MV0_I01_2_1_r', 'i_t_MV0_I01_2_1_i', 'i_t_MV0_I01_2_2_r', 'i_t_MV0_I01_2_2_i', 'i_t_MV0_I01_2_3_r', 'i_t_MV0_I01_2_3_i', 'i_t_MV0_C01_1_0_r', 'i_t_MV0_C01_1_0_i', 'i_t_MV0_C01_1_1_r', 'i_t_MV0_C01_1_1_i', 'i_t_MV0_C01_1_2_r', 'i_t_MV0_C01_1_2_i', 'i_t_MV0_C01_2_0_r', 'i_t_MV0_C01_2_0_i', 'i_t_MV0_C01_2_1_r', 'i_t_MV0_C01_2_1_i', 'i_t_MV0_C01_2_2_r', 'i_t_MV0_C01_2_2_i', 'i_t_MV0_C01_2_3_r', 'i_t_MV0_C01_2_3_i', 'i_vsc_I01_a_m', 'i_vsc_I01_b_m', 'i_vsc_I01_c_m', 'i_vsc_I01_n_m', 'p_vsc_I01', 'p_vsc_loss_I01', 'i_vsc_R01_a_m', 'i_vsc_R01_b_m', 'i_vsc_R01_c_m', 'i_vsc_R01_n_m', 'p_vsc_R01', 'p_vsc_loss_R01', 'v_dc_S01', 'i_vsc_C01_a_m', 'i_vsc_C01_b_m', 'i_vsc_C01_c_m', 'i_vsc_C01_n_m', 'p_vsc_C01', 'p_vsc_loss_C01', 'v_dc_D01', 'i_vsc_R10_a_m', 'i_vsc_R10_b_m', 'i_vsc_R10_c_m', 'i_vsc_R10_n_m', 'p_vsc_R10', 'p_vsc_loss_R10', 'v_dc_S10', 'i_vsc_R14_a_m', 'i_vsc_R14_b_m', 'i_vsc_R14_c_m', 'i_vsc_R14_n_m', 'p_vsc_R14', 'p_vsc_loss_R14', 'v_dc_S14', 'i_vsc_I02_a_m', 'i_vsc_I02_b_m', 'i_vsc_I02_c_m', 'i_vsc_I02_n_m', 'p_vsc_I02', 'p_vsc_loss_I02', 'v_dc_H02', 'i_vsc_C09_a_m', 'i_vsc_C09_b_m', 'i_vsc_C09_c_m', 'i_vsc_C09_n_m', 'p_vsc_C09', 'p_vsc_loss_C09', 'v_dc_D09', 'i_vsc_C11_a_m', 'i_vsc_C11_b_m', 'i_vsc_C11_c_m', 'i_vsc_C11_n_m', 'p_vsc_C11', 'p_vsc_loss_C11', 'v_dc_D11', 'i_vsc_C16_a_m', 'i_vsc_C16_b_m', 'i_vsc_C16_c_m', 'i_vsc_C16_n_m', 'p_vsc_C16', 'p_vsc_loss_C16', 'v_dc_D16', 'i_vsc_MV0_a_m', 'i_vsc_MV0_b_m', 'i_vsc_MV0_c_m', 'p_MV0', 'q_MV0', 'xi_freq', 'u_freq'] 
        self.x_list = ['xi_freq', 'p_vsc_a_R01', 'p_vsc_b_R01', 'p_vsc_c_R01', 'p_vsc_a_R10', 'p_vsc_b_R10', 'p_vsc_c_R10', 'p_vsc_a_R14', 'p_vsc_b_R14', 'p_vsc_c_R14', 'p_vsc_a_I02', 'p_vsc_b_I02', 'p_vsc_c_I02', 'p_vsc_a_C01', 'p_vsc_b_C01', 'p_vsc_c_C01', 'p_vsc_a_C09', 'p_vsc_b_C09', 'p_vsc_c_C09', 'p_vsc_a_C11', 'p_vsc_b_C11', 'p_vsc_c_C11', 'p_vsc_a_C16', 'p_vsc_b_C16', 'p_vsc_c_C16'] 
        self.y_run_list = ['V_MV0_0_r', 'V_MV0_0_i', 'V_MV0_1_r', 'V_MV0_1_i', 'V_MV0_2_r', 'V_MV0_2_i', 'V_R01_0_r', 'V_R01_0_i', 'V_R01_1_r', 'V_R01_1_i', 'V_R01_2_r', 'V_R01_2_i', 'V_R01_3_r', 'V_R01_3_i', 'V_R02_0_r', 'V_R02_0_i', 'V_R02_1_r', 'V_R02_1_i', 'V_R02_2_r', 'V_R02_2_i', 'V_R02_3_r', 'V_R02_3_i', 'V_R03_0_r', 'V_R03_0_i', 'V_R03_1_r', 'V_R03_1_i', 'V_R03_2_r', 'V_R03_2_i', 'V_R03_3_r', 'V_R03_3_i', 'V_R04_0_r', 'V_R04_0_i', 'V_R04_1_r', 'V_R04_1_i', 'V_R04_2_r', 'V_R04_2_i', 'V_R04_3_r', 'V_R04_3_i', 'V_R05_0_r', 'V_R05_0_i', 'V_R05_1_r', 'V_R05_1_i', 'V_R05_2_r', 'V_R05_2_i', 'V_R05_3_r', 'V_R05_3_i', 'V_R06_0_r', 'V_R06_0_i', 'V_R06_1_r', 'V_R06_1_i', 'V_R06_2_r', 'V_R06_2_i', 'V_R06_3_r', 'V_R06_3_i', 'V_R07_0_r', 'V_R07_0_i', 'V_R07_1_r', 'V_R07_1_i', 'V_R07_2_r', 'V_R07_2_i', 'V_R07_3_r', 'V_R07_3_i', 'V_R08_0_r', 'V_R08_0_i', 'V_R08_1_r', 'V_R08_1_i', 'V_R08_2_r', 'V_R08_2_i', 'V_R08_3_r', 'V_R08_3_i', 'V_R09_0_r', 'V_R09_0_i', 'V_R09_1_r', 'V_R09_1_i', 'V_R09_2_r', 'V_R09_2_i', 'V_R09_3_r', 'V_R09_3_i', 'V_R10_0_r', 'V_R10_0_i', 'V_R10_1_r', 'V_R10_1_i', 'V_R10_2_r', 'V_R10_2_i', 'V_R10_3_r', 'V_R10_3_i', 'V_R11_0_r', 'V_R11_0_i', 'V_R11_1_r', 'V_R11_1_i', 'V_R11_2_r', 'V_R11_2_i', 'V_R11_3_r', 'V_R11_3_i', 'V_R12_0_r', 'V_R12_0_i', 'V_R12_1_r', 'V_R12_1_i', 'V_R12_2_r', 'V_R12_2_i', 'V_R12_3_r', 'V_R12_3_i', 'V_R13_0_r', 'V_R13_0_i', 'V_R13_1_r', 'V_R13_1_i', 'V_R13_2_r', 'V_R13_2_i', 'V_R13_3_r', 'V_R13_3_i', 'V_R14_0_r', 'V_R14_0_i', 'V_R14_1_r', 'V_R14_1_i', 'V_R14_2_r', 'V_R14_2_i', 'V_R14_3_r', 'V_R14_3_i', 'V_R15_0_r', 'V_R15_0_i', 'V_R15_1_r', 'V_R15_1_i', 'V_R15_2_r', 'V_R15_2_i', 'V_R15_3_r', 'V_R15_3_i', 'V_R16_0_r', 'V_R16_0_i', 'V_R16_1_r', 'V_R16_1_i', 'V_R16_2_r', 'V_R16_2_i', 'V_R16_3_r', 'V_R16_3_i', 'V_R17_0_r', 'V_R17_0_i', 'V_R17_1_r', 'V_R17_1_i', 'V_R17_2_r', 'V_R17_2_i', 'V_R17_3_r', 'V_R17_3_i', 'V_R18_0_r', 'V_R18_0_i', 'V_R18_1_r', 'V_R18_1_i', 'V_R18_2_r', 'V_R18_2_i', 'V_R18_3_r', 'V_R18_3_i', 'V_I01_0_r', 'V_I01_0_i', 'V_I01_1_r', 'V_I01_1_i', 'V_I01_2_r', 'V_I01_2_i', 'V_I01_3_r', 'V_I01_3_i', 'V_I02_0_r', 'V_I02_0_i', 'V_I02_1_r', 'V_I02_1_i', 'V_I02_2_r', 'V_I02_2_i', 'V_I02_3_r', 'V_I02_3_i', 'V_C01_0_r', 'V_C01_0_i', 'V_C01_1_r', 'V_C01_1_i', 'V_C01_2_r', 'V_C01_2_i', 'V_C01_3_r', 'V_C01_3_i', 'V_C02_0_r', 'V_C02_0_i', 'V_C02_1_r', 'V_C02_1_i', 'V_C02_2_r', 'V_C02_2_i', 'V_C02_3_r', 'V_C02_3_i', 'V_C03_0_r', 'V_C03_0_i', 'V_C03_1_r', 'V_C03_1_i', 'V_C03_2_r', 'V_C03_2_i', 'V_C03_3_r', 'V_C03_3_i', 'V_C04_0_r', 'V_C04_0_i', 'V_C04_1_r', 'V_C04_1_i', 'V_C04_2_r', 'V_C04_2_i', 'V_C04_3_r', 'V_C04_3_i', 'V_C05_0_r', 'V_C05_0_i', 'V_C05_1_r', 'V_C05_1_i', 'V_C05_2_r', 'V_C05_2_i', 'V_C05_3_r', 'V_C05_3_i', 'V_C06_0_r', 'V_C06_0_i', 'V_C06_1_r', 'V_C06_1_i', 'V_C06_2_r', 'V_C06_2_i', 'V_C06_3_r', 'V_C06_3_i', 'V_C07_0_r', 'V_C07_0_i', 'V_C07_1_r', 'V_C07_1_i', 'V_C07_2_r', 'V_C07_2_i', 'V_C07_3_r', 'V_C07_3_i', 'V_C08_0_r', 'V_C08_0_i', 'V_C08_1_r', 'V_C08_1_i', 'V_C08_2_r', 'V_C08_2_i', 'V_C08_3_r', 'V_C08_3_i', 'V_C09_0_r', 'V_C09_0_i', 'V_C09_1_r', 'V_C09_1_i', 'V_C09_2_r', 'V_C09_2_i', 'V_C09_3_r', 'V_C09_3_i', 'V_C10_0_r', 'V_C10_0_i', 'V_C10_1_r', 'V_C10_1_i', 'V_C10_2_r', 'V_C10_2_i', 'V_C10_3_r', 'V_C10_3_i', 'V_C11_0_r', 'V_C11_0_i', 'V_C11_1_r', 'V_C11_1_i', 'V_C11_2_r', 'V_C11_2_i', 'V_C11_3_r', 'V_C11_3_i', 'V_C12_0_r', 'V_C12_0_i', 'V_C12_1_r', 'V_C12_1_i', 'V_C12_2_r', 'V_C12_2_i', 'V_C12_3_r', 'V_C12_3_i', 'V_C13_0_r', 'V_C13_0_i', 'V_C13_1_r', 'V_C13_1_i', 'V_C13_2_r', 'V_C13_2_i', 'V_C13_3_r', 'V_C13_3_i', 'V_C14_0_r', 'V_C14_0_i', 'V_C14_1_r', 'V_C14_1_i', 'V_C14_2_r', 'V_C14_2_i', 'V_C14_3_r', 'V_C14_3_i', 'V_C15_0_r', 'V_C15_0_i', 'V_C15_1_r', 'V_C15_1_i', 'V_C15_2_r', 'V_C15_2_i', 'V_C15_3_r', 'V_C15_3_i', 'V_C16_0_r', 'V_C16_0_i', 'V_C16_1_r', 'V_C16_1_i', 'V_C16_2_r', 'V_C16_2_i', 'V_C16_3_r', 'V_C16_3_i', 'V_C17_0_r', 'V_C17_0_i', 'V_C17_1_r', 'V_C17_1_i', 'V_C17_2_r', 'V_C17_2_i', 'V_C17_3_r', 'V_C17_3_i', 'V_C18_0_r', 'V_C18_0_i', 'V_C18_1_r', 'V_C18_1_i', 'V_C18_2_r', 'V_C18_2_i', 'V_C18_3_r', 'V_C18_3_i', 'V_C19_0_r', 'V_C19_0_i', 'V_C19_1_r', 'V_C19_1_i', 'V_C19_2_r', 'V_C19_2_i', 'V_C19_3_r', 'V_C19_3_i', 'V_C20_0_r', 'V_C20_0_i', 'V_C20_1_r', 'V_C20_1_i', 'V_C20_2_r', 'V_C20_2_i', 'V_C20_3_r', 'V_C20_3_i', 'V_S01_0_r', 'V_S01_0_i', 'V_S01_1_r', 'V_S01_1_i', 'V_S03_0_r', 'V_S03_0_i', 'V_S03_1_r', 'V_S03_1_i', 'V_S04_0_r', 'V_S04_0_i', 'V_S04_1_r', 'V_S04_1_i', 'V_S06_0_r', 'V_S06_0_i', 'V_S06_1_r', 'V_S06_1_i', 'V_S07_0_r', 'V_S07_0_i', 'V_S07_1_r', 'V_S07_1_i', 'V_S09_0_r', 'V_S09_0_i', 'V_S09_1_r', 'V_S09_1_i', 'V_S10_0_r', 'V_S10_0_i', 'V_S10_1_r', 'V_S10_1_i', 'V_S11_0_r', 'V_S11_0_i', 'V_S11_1_r', 'V_S11_1_i', 'V_S16_0_r', 'V_S16_0_i', 'V_S16_1_r', 'V_S16_1_i', 'V_S17_0_r', 'V_S17_0_i', 'V_S17_1_r', 'V_S17_1_i', 'V_S18_0_r', 'V_S18_0_i', 'V_S18_1_r', 'V_S18_1_i', 'V_S14_0_r', 'V_S14_0_i', 'V_S14_1_r', 'V_S14_1_i', 'V_S15_0_r', 'V_S15_0_i', 'V_S15_1_r', 'V_S15_1_i', 'V_H01_0_r', 'V_H01_0_i', 'V_H01_1_r', 'V_H01_1_i', 'V_H02_0_r', 'V_H02_0_i', 'V_H02_1_r', 'V_H02_1_i', 'V_D01_0_r', 'V_D01_0_i', 'V_D01_1_r', 'V_D01_1_i', 'V_D03_0_r', 'V_D03_0_i', 'V_D03_1_r', 'V_D03_1_i', 'V_D05_0_r', 'V_D05_0_i', 'V_D05_1_r', 'V_D05_1_i', 'V_D08_0_r', 'V_D08_0_i', 'V_D08_1_r', 'V_D08_1_i', 'V_D09_0_r', 'V_D09_0_i', 'V_D09_1_r', 'V_D09_1_i', 'V_D11_0_r', 'V_D11_0_i', 'V_D11_1_r', 'V_D11_1_i', 'V_D12_0_r', 'V_D12_0_i', 'V_D12_1_r', 'V_D12_1_i', 'V_D16_0_r', 'V_D16_0_i', 'V_D16_1_r', 'V_D16_1_i', 'V_D17_0_r', 'V_D17_0_i', 'V_D17_1_r', 'V_D17_1_i', 'V_D19_0_r', 'V_D19_0_i', 'V_D19_1_r', 'V_D19_1_i', 'V_D20_0_r', 'V_D20_0_i', 'V_D20_1_r', 'V_D20_1_i', 'i_load_R01_a_r', 'i_load_R01_a_i', 'i_load_R01_b_r', 'i_load_R01_b_i', 'i_load_R01_c_r', 'i_load_R01_c_i', 'i_load_R01_n_r', 'i_load_R01_n_i', 'i_load_R11_a_r', 'i_load_R11_a_i', 'i_load_R11_b_r', 'i_load_R11_b_i', 'i_load_R11_c_r', 'i_load_R11_c_i', 'i_load_R11_n_r', 'i_load_R11_n_i', 'i_load_R15_a_r', 'i_load_R15_a_i', 'i_load_R15_b_r', 'i_load_R15_b_i', 'i_load_R15_c_r', 'i_load_R15_c_i', 'i_load_R15_n_r', 'i_load_R15_n_i', 'i_load_R16_a_r', 'i_load_R16_a_i', 'i_load_R16_b_r', 'i_load_R16_b_i', 'i_load_R16_c_r', 'i_load_R16_c_i', 'i_load_R16_n_r', 'i_load_R16_n_i', 'i_load_R17_a_r', 'i_load_R17_a_i', 'i_load_R17_b_r', 'i_load_R17_b_i', 'i_load_R17_c_r', 'i_load_R17_c_i', 'i_load_R17_n_r', 'i_load_R17_n_i', 'i_load_R18_a_r', 'i_load_R18_a_i', 'i_load_R18_b_r', 'i_load_R18_b_i', 'i_load_R18_c_r', 'i_load_R18_c_i', 'i_load_R18_n_r', 'i_load_R18_n_i', 'i_load_I02_a_r', 'i_load_I02_a_i', 'i_load_I02_b_r', 'i_load_I02_b_i', 'i_load_I02_c_r', 'i_load_I02_c_i', 'i_load_I02_n_r', 'i_load_I02_n_i', 'i_load_C01_a_r', 'i_load_C01_a_i', 'i_load_C01_b_r', 'i_load_C01_b_i', 'i_load_C01_c_r', 'i_load_C01_c_i', 'i_load_C01_n_r', 'i_load_C01_n_i', 'i_load_C12_a_r', 'i_load_C12_a_i', 'i_load_C12_b_r', 'i_load_C12_b_i', 'i_load_C12_c_r', 'i_load_C12_c_i', 'i_load_C12_n_r', 'i_load_C12_n_i', 'i_load_C13_a_r', 'i_load_C13_a_i', 'i_load_C13_b_r', 'i_load_C13_b_i', 'i_load_C13_c_r', 'i_load_C13_c_i', 'i_load_C13_n_r', 'i_load_C13_n_i', 'i_load_C14_a_r', 'i_load_C14_a_i', 'i_load_C14_b_r', 'i_load_C14_b_i', 'i_load_C14_c_r', 'i_load_C14_c_i', 'i_load_C14_n_r', 'i_load_C14_n_i', 'i_load_C17_a_r', 'i_load_C17_a_i', 'i_load_C17_b_r', 'i_load_C17_b_i', 'i_load_C17_c_r', 'i_load_C17_c_i', 'i_load_C17_n_r', 'i_load_C17_n_i', 'i_load_C18_a_r', 'i_load_C18_a_i', 'i_load_C18_b_r', 'i_load_C18_b_i', 'i_load_C18_c_r', 'i_load_C18_c_i', 'i_load_C18_n_r', 'i_load_C18_n_i', 'i_load_C19_a_r', 'i_load_C19_a_i', 'i_load_C19_b_r', 'i_load_C19_b_i', 'i_load_C19_c_r', 'i_load_C19_c_i', 'i_load_C19_n_r', 'i_load_C19_n_i', 'i_load_C20_a_r', 'i_load_C20_a_i', 'i_load_C20_b_r', 'i_load_C20_b_i', 'i_load_C20_c_r', 'i_load_C20_c_i', 'i_load_C20_n_r', 'i_load_C20_n_i', 'i_load_S15_p_r', 'i_load_S11_p_r', 'i_load_S16_p_r', 'i_load_S17_p_r', 'i_load_S18_p_r', 'i_load_H02_p_r', 'i_load_D12_p_r', 'i_load_D17_p_r', 'i_load_D19_p_r', 'i_load_D20_p_r', 'p_a_d_I01', 'p_b_d_I01', 'p_c_d_I01', 'p_n_d_I01', 'i_vsc_I01_a_r', 'i_vsc_I01_a_i', 'i_vsc_I01_b_r', 'i_vsc_I01_b_i', 'i_vsc_I01_c_r', 'i_vsc_I01_c_i', 'i_vsc_I01_n_r', 'i_vsc_I01_n_i', 'i_vsc_pos_H01_sp', 'i_vsc_H01_sn', 'v_og_H01', 'p_vsc_H01', 'i_vsc_R01_a_r', 'i_vsc_R01_a_i', 'i_vsc_R01_b_r', 'i_vsc_R01_b_i', 'i_vsc_R01_c_r', 'i_vsc_R01_c_i', 'i_vsc_R01_n_r', 'i_vsc_R01_n_i', 'i_vsc_pos_S01_sp', 'i_vsc_S01_sn', 'p_vsc_S01', 'i_vsc_C01_a_r', 'i_vsc_C01_a_i', 'i_vsc_C01_b_r', 'i_vsc_C01_b_i', 'i_vsc_C01_c_r', 'i_vsc_C01_c_i', 'i_vsc_C01_n_r', 'i_vsc_C01_n_i', 'i_vsc_pos_D01_sp', 'i_vsc_D01_sn', 'p_vsc_D01', 'i_vsc_R10_a_r', 'i_vsc_R10_a_i', 'i_vsc_R10_b_r', 'i_vsc_R10_b_i', 'i_vsc_R10_c_r', 'i_vsc_R10_c_i', 'i_vsc_R10_n_r', 'i_vsc_R10_n_i', 'i_vsc_pos_S10_sp', 'i_vsc_S10_sn', 'p_vsc_S10', 'i_vsc_R14_a_r', 'i_vsc_R14_a_i', 'i_vsc_R14_b_r', 'i_vsc_R14_b_i', 'i_vsc_R14_c_r', 'i_vsc_R14_c_i', 'i_vsc_R14_n_r', 'i_vsc_R14_n_i', 'i_vsc_pos_S14_sp', 'i_vsc_S14_sn', 'p_vsc_S14', 'i_vsc_I02_a_r', 'i_vsc_I02_a_i', 'i_vsc_I02_b_r', 'i_vsc_I02_b_i', 'i_vsc_I02_c_r', 'i_vsc_I02_c_i', 'i_vsc_I02_n_r', 'i_vsc_I02_n_i', 'i_vsc_pos_H02_sp', 'i_vsc_H02_sn', 'p_vsc_H02', 'i_vsc_C09_a_r', 'i_vsc_C09_a_i', 'i_vsc_C09_b_r', 'i_vsc_C09_b_i', 'i_vsc_C09_c_r', 'i_vsc_C09_c_i', 'i_vsc_C09_n_r', 'i_vsc_C09_n_i', 'i_vsc_pos_D09_sp', 'i_vsc_D09_sn', 'p_vsc_D09', 'i_vsc_C11_a_r', 'i_vsc_C11_a_i', 'i_vsc_C11_b_r', 'i_vsc_C11_b_i', 'i_vsc_C11_c_r', 'i_vsc_C11_c_i', 'i_vsc_C11_n_r', 'i_vsc_C11_n_i', 'i_vsc_pos_D11_sp', 'i_vsc_D11_sn', 'p_vsc_D11', 'i_vsc_C16_a_r', 'i_vsc_C16_a_i', 'i_vsc_C16_b_r', 'i_vsc_C16_b_i', 'i_vsc_C16_c_r', 'i_vsc_C16_c_i', 'i_vsc_C16_n_r', 'i_vsc_C16_n_i', 'i_vsc_pos_D16_sp', 'i_vsc_D16_sn', 'p_vsc_D16', 'i_vsc_MV0_a_r', 'i_vsc_MV0_b_r', 'i_vsc_MV0_c_r', 'i_vsc_MV0_a_i', 'i_vsc_MV0_b_i', 'i_vsc_MV0_c_i', 'omega_coi', 'p_agc'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['V_MV0_0_r', 'V_MV0_0_i', 'V_MV0_1_r', 'V_MV0_1_i', 'V_MV0_2_r', 'V_MV0_2_i', 'V_R01_0_r', 'V_R01_0_i', 'V_R01_1_r', 'V_R01_1_i', 'V_R01_2_r', 'V_R01_2_i', 'V_R01_3_r', 'V_R01_3_i', 'V_R02_0_r', 'V_R02_0_i', 'V_R02_1_r', 'V_R02_1_i', 'V_R02_2_r', 'V_R02_2_i', 'V_R02_3_r', 'V_R02_3_i', 'V_R03_0_r', 'V_R03_0_i', 'V_R03_1_r', 'V_R03_1_i', 'V_R03_2_r', 'V_R03_2_i', 'V_R03_3_r', 'V_R03_3_i', 'V_R04_0_r', 'V_R04_0_i', 'V_R04_1_r', 'V_R04_1_i', 'V_R04_2_r', 'V_R04_2_i', 'V_R04_3_r', 'V_R04_3_i', 'V_R05_0_r', 'V_R05_0_i', 'V_R05_1_r', 'V_R05_1_i', 'V_R05_2_r', 'V_R05_2_i', 'V_R05_3_r', 'V_R05_3_i', 'V_R06_0_r', 'V_R06_0_i', 'V_R06_1_r', 'V_R06_1_i', 'V_R06_2_r', 'V_R06_2_i', 'V_R06_3_r', 'V_R06_3_i', 'V_R07_0_r', 'V_R07_0_i', 'V_R07_1_r', 'V_R07_1_i', 'V_R07_2_r', 'V_R07_2_i', 'V_R07_3_r', 'V_R07_3_i', 'V_R08_0_r', 'V_R08_0_i', 'V_R08_1_r', 'V_R08_1_i', 'V_R08_2_r', 'V_R08_2_i', 'V_R08_3_r', 'V_R08_3_i', 'V_R09_0_r', 'V_R09_0_i', 'V_R09_1_r', 'V_R09_1_i', 'V_R09_2_r', 'V_R09_2_i', 'V_R09_3_r', 'V_R09_3_i', 'V_R10_0_r', 'V_R10_0_i', 'V_R10_1_r', 'V_R10_1_i', 'V_R10_2_r', 'V_R10_2_i', 'V_R10_3_r', 'V_R10_3_i', 'V_R11_0_r', 'V_R11_0_i', 'V_R11_1_r', 'V_R11_1_i', 'V_R11_2_r', 'V_R11_2_i', 'V_R11_3_r', 'V_R11_3_i', 'V_R12_0_r', 'V_R12_0_i', 'V_R12_1_r', 'V_R12_1_i', 'V_R12_2_r', 'V_R12_2_i', 'V_R12_3_r', 'V_R12_3_i', 'V_R13_0_r', 'V_R13_0_i', 'V_R13_1_r', 'V_R13_1_i', 'V_R13_2_r', 'V_R13_2_i', 'V_R13_3_r', 'V_R13_3_i', 'V_R14_0_r', 'V_R14_0_i', 'V_R14_1_r', 'V_R14_1_i', 'V_R14_2_r', 'V_R14_2_i', 'V_R14_3_r', 'V_R14_3_i', 'V_R15_0_r', 'V_R15_0_i', 'V_R15_1_r', 'V_R15_1_i', 'V_R15_2_r', 'V_R15_2_i', 'V_R15_3_r', 'V_R15_3_i', 'V_R16_0_r', 'V_R16_0_i', 'V_R16_1_r', 'V_R16_1_i', 'V_R16_2_r', 'V_R16_2_i', 'V_R16_3_r', 'V_R16_3_i', 'V_R17_0_r', 'V_R17_0_i', 'V_R17_1_r', 'V_R17_1_i', 'V_R17_2_r', 'V_R17_2_i', 'V_R17_3_r', 'V_R17_3_i', 'V_R18_0_r', 'V_R18_0_i', 'V_R18_1_r', 'V_R18_1_i', 'V_R18_2_r', 'V_R18_2_i', 'V_R18_3_r', 'V_R18_3_i', 'V_I01_0_r', 'V_I01_0_i', 'V_I01_1_r', 'V_I01_1_i', 'V_I01_2_r', 'V_I01_2_i', 'V_I01_3_r', 'V_I01_3_i', 'V_I02_0_r', 'V_I02_0_i', 'V_I02_1_r', 'V_I02_1_i', 'V_I02_2_r', 'V_I02_2_i', 'V_I02_3_r', 'V_I02_3_i', 'V_C01_0_r', 'V_C01_0_i', 'V_C01_1_r', 'V_C01_1_i', 'V_C01_2_r', 'V_C01_2_i', 'V_C01_3_r', 'V_C01_3_i', 'V_C02_0_r', 'V_C02_0_i', 'V_C02_1_r', 'V_C02_1_i', 'V_C02_2_r', 'V_C02_2_i', 'V_C02_3_r', 'V_C02_3_i', 'V_C03_0_r', 'V_C03_0_i', 'V_C03_1_r', 'V_C03_1_i', 'V_C03_2_r', 'V_C03_2_i', 'V_C03_3_r', 'V_C03_3_i', 'V_C04_0_r', 'V_C04_0_i', 'V_C04_1_r', 'V_C04_1_i', 'V_C04_2_r', 'V_C04_2_i', 'V_C04_3_r', 'V_C04_3_i', 'V_C05_0_r', 'V_C05_0_i', 'V_C05_1_r', 'V_C05_1_i', 'V_C05_2_r', 'V_C05_2_i', 'V_C05_3_r', 'V_C05_3_i', 'V_C06_0_r', 'V_C06_0_i', 'V_C06_1_r', 'V_C06_1_i', 'V_C06_2_r', 'V_C06_2_i', 'V_C06_3_r', 'V_C06_3_i', 'V_C07_0_r', 'V_C07_0_i', 'V_C07_1_r', 'V_C07_1_i', 'V_C07_2_r', 'V_C07_2_i', 'V_C07_3_r', 'V_C07_3_i', 'V_C08_0_r', 'V_C08_0_i', 'V_C08_1_r', 'V_C08_1_i', 'V_C08_2_r', 'V_C08_2_i', 'V_C08_3_r', 'V_C08_3_i', 'V_C09_0_r', 'V_C09_0_i', 'V_C09_1_r', 'V_C09_1_i', 'V_C09_2_r', 'V_C09_2_i', 'V_C09_3_r', 'V_C09_3_i', 'V_C10_0_r', 'V_C10_0_i', 'V_C10_1_r', 'V_C10_1_i', 'V_C10_2_r', 'V_C10_2_i', 'V_C10_3_r', 'V_C10_3_i', 'V_C11_0_r', 'V_C11_0_i', 'V_C11_1_r', 'V_C11_1_i', 'V_C11_2_r', 'V_C11_2_i', 'V_C11_3_r', 'V_C11_3_i', 'V_C12_0_r', 'V_C12_0_i', 'V_C12_1_r', 'V_C12_1_i', 'V_C12_2_r', 'V_C12_2_i', 'V_C12_3_r', 'V_C12_3_i', 'V_C13_0_r', 'V_C13_0_i', 'V_C13_1_r', 'V_C13_1_i', 'V_C13_2_r', 'V_C13_2_i', 'V_C13_3_r', 'V_C13_3_i', 'V_C14_0_r', 'V_C14_0_i', 'V_C14_1_r', 'V_C14_1_i', 'V_C14_2_r', 'V_C14_2_i', 'V_C14_3_r', 'V_C14_3_i', 'V_C15_0_r', 'V_C15_0_i', 'V_C15_1_r', 'V_C15_1_i', 'V_C15_2_r', 'V_C15_2_i', 'V_C15_3_r', 'V_C15_3_i', 'V_C16_0_r', 'V_C16_0_i', 'V_C16_1_r', 'V_C16_1_i', 'V_C16_2_r', 'V_C16_2_i', 'V_C16_3_r', 'V_C16_3_i', 'V_C17_0_r', 'V_C17_0_i', 'V_C17_1_r', 'V_C17_1_i', 'V_C17_2_r', 'V_C17_2_i', 'V_C17_3_r', 'V_C17_3_i', 'V_C18_0_r', 'V_C18_0_i', 'V_C18_1_r', 'V_C18_1_i', 'V_C18_2_r', 'V_C18_2_i', 'V_C18_3_r', 'V_C18_3_i', 'V_C19_0_r', 'V_C19_0_i', 'V_C19_1_r', 'V_C19_1_i', 'V_C19_2_r', 'V_C19_2_i', 'V_C19_3_r', 'V_C19_3_i', 'V_C20_0_r', 'V_C20_0_i', 'V_C20_1_r', 'V_C20_1_i', 'V_C20_2_r', 'V_C20_2_i', 'V_C20_3_r', 'V_C20_3_i', 'V_S01_0_r', 'V_S01_0_i', 'V_S01_1_r', 'V_S01_1_i', 'V_S03_0_r', 'V_S03_0_i', 'V_S03_1_r', 'V_S03_1_i', 'V_S04_0_r', 'V_S04_0_i', 'V_S04_1_r', 'V_S04_1_i', 'V_S06_0_r', 'V_S06_0_i', 'V_S06_1_r', 'V_S06_1_i', 'V_S07_0_r', 'V_S07_0_i', 'V_S07_1_r', 'V_S07_1_i', 'V_S09_0_r', 'V_S09_0_i', 'V_S09_1_r', 'V_S09_1_i', 'V_S10_0_r', 'V_S10_0_i', 'V_S10_1_r', 'V_S10_1_i', 'V_S11_0_r', 'V_S11_0_i', 'V_S11_1_r', 'V_S11_1_i', 'V_S16_0_r', 'V_S16_0_i', 'V_S16_1_r', 'V_S16_1_i', 'V_S17_0_r', 'V_S17_0_i', 'V_S17_1_r', 'V_S17_1_i', 'V_S18_0_r', 'V_S18_0_i', 'V_S18_1_r', 'V_S18_1_i', 'V_S14_0_r', 'V_S14_0_i', 'V_S14_1_r', 'V_S14_1_i', 'V_S15_0_r', 'V_S15_0_i', 'V_S15_1_r', 'V_S15_1_i', 'V_H01_0_r', 'V_H01_0_i', 'V_H01_1_r', 'V_H01_1_i', 'V_H02_0_r', 'V_H02_0_i', 'V_H02_1_r', 'V_H02_1_i', 'V_D01_0_r', 'V_D01_0_i', 'V_D01_1_r', 'V_D01_1_i', 'V_D03_0_r', 'V_D03_0_i', 'V_D03_1_r', 'V_D03_1_i', 'V_D05_0_r', 'V_D05_0_i', 'V_D05_1_r', 'V_D05_1_i', 'V_D08_0_r', 'V_D08_0_i', 'V_D08_1_r', 'V_D08_1_i', 'V_D09_0_r', 'V_D09_0_i', 'V_D09_1_r', 'V_D09_1_i', 'V_D11_0_r', 'V_D11_0_i', 'V_D11_1_r', 'V_D11_1_i', 'V_D12_0_r', 'V_D12_0_i', 'V_D12_1_r', 'V_D12_1_i', 'V_D16_0_r', 'V_D16_0_i', 'V_D16_1_r', 'V_D16_1_i', 'V_D17_0_r', 'V_D17_0_i', 'V_D17_1_r', 'V_D17_1_i', 'V_D19_0_r', 'V_D19_0_i', 'V_D19_1_r', 'V_D19_1_i', 'V_D20_0_r', 'V_D20_0_i', 'V_D20_1_r', 'V_D20_1_i', 'i_load_R01_a_r', 'i_load_R01_a_i', 'i_load_R01_b_r', 'i_load_R01_b_i', 'i_load_R01_c_r', 'i_load_R01_c_i', 'i_load_R01_n_r', 'i_load_R01_n_i', 'i_load_R11_a_r', 'i_load_R11_a_i', 'i_load_R11_b_r', 'i_load_R11_b_i', 'i_load_R11_c_r', 'i_load_R11_c_i', 'i_load_R11_n_r', 'i_load_R11_n_i', 'i_load_R15_a_r', 'i_load_R15_a_i', 'i_load_R15_b_r', 'i_load_R15_b_i', 'i_load_R15_c_r', 'i_load_R15_c_i', 'i_load_R15_n_r', 'i_load_R15_n_i', 'i_load_R16_a_r', 'i_load_R16_a_i', 'i_load_R16_b_r', 'i_load_R16_b_i', 'i_load_R16_c_r', 'i_load_R16_c_i', 'i_load_R16_n_r', 'i_load_R16_n_i', 'i_load_R17_a_r', 'i_load_R17_a_i', 'i_load_R17_b_r', 'i_load_R17_b_i', 'i_load_R17_c_r', 'i_load_R17_c_i', 'i_load_R17_n_r', 'i_load_R17_n_i', 'i_load_R18_a_r', 'i_load_R18_a_i', 'i_load_R18_b_r', 'i_load_R18_b_i', 'i_load_R18_c_r', 'i_load_R18_c_i', 'i_load_R18_n_r', 'i_load_R18_n_i', 'i_load_I02_a_r', 'i_load_I02_a_i', 'i_load_I02_b_r', 'i_load_I02_b_i', 'i_load_I02_c_r', 'i_load_I02_c_i', 'i_load_I02_n_r', 'i_load_I02_n_i', 'i_load_C01_a_r', 'i_load_C01_a_i', 'i_load_C01_b_r', 'i_load_C01_b_i', 'i_load_C01_c_r', 'i_load_C01_c_i', 'i_load_C01_n_r', 'i_load_C01_n_i', 'i_load_C12_a_r', 'i_load_C12_a_i', 'i_load_C12_b_r', 'i_load_C12_b_i', 'i_load_C12_c_r', 'i_load_C12_c_i', 'i_load_C12_n_r', 'i_load_C12_n_i', 'i_load_C13_a_r', 'i_load_C13_a_i', 'i_load_C13_b_r', 'i_load_C13_b_i', 'i_load_C13_c_r', 'i_load_C13_c_i', 'i_load_C13_n_r', 'i_load_C13_n_i', 'i_load_C14_a_r', 'i_load_C14_a_i', 'i_load_C14_b_r', 'i_load_C14_b_i', 'i_load_C14_c_r', 'i_load_C14_c_i', 'i_load_C14_n_r', 'i_load_C14_n_i', 'i_load_C17_a_r', 'i_load_C17_a_i', 'i_load_C17_b_r', 'i_load_C17_b_i', 'i_load_C17_c_r', 'i_load_C17_c_i', 'i_load_C17_n_r', 'i_load_C17_n_i', 'i_load_C18_a_r', 'i_load_C18_a_i', 'i_load_C18_b_r', 'i_load_C18_b_i', 'i_load_C18_c_r', 'i_load_C18_c_i', 'i_load_C18_n_r', 'i_load_C18_n_i', 'i_load_C19_a_r', 'i_load_C19_a_i', 'i_load_C19_b_r', 'i_load_C19_b_i', 'i_load_C19_c_r', 'i_load_C19_c_i', 'i_load_C19_n_r', 'i_load_C19_n_i', 'i_load_C20_a_r', 'i_load_C20_a_i', 'i_load_C20_b_r', 'i_load_C20_b_i', 'i_load_C20_c_r', 'i_load_C20_c_i', 'i_load_C20_n_r', 'i_load_C20_n_i', 'i_load_S15_p_r', 'i_load_S11_p_r', 'i_load_S16_p_r', 'i_load_S17_p_r', 'i_load_S18_p_r', 'i_load_H02_p_r', 'i_load_D12_p_r', 'i_load_D17_p_r', 'i_load_D19_p_r', 'i_load_D20_p_r', 'p_a_d_I01', 'p_b_d_I01', 'p_c_d_I01', 'p_n_d_I01', 'i_vsc_I01_a_r', 'i_vsc_I01_a_i', 'i_vsc_I01_b_r', 'i_vsc_I01_b_i', 'i_vsc_I01_c_r', 'i_vsc_I01_c_i', 'i_vsc_I01_n_r', 'i_vsc_I01_n_i', 'i_vsc_pos_H01_sp', 'i_vsc_H01_sn', 'v_og_H01', 'p_vsc_H01', 'i_vsc_R01_a_r', 'i_vsc_R01_a_i', 'i_vsc_R01_b_r', 'i_vsc_R01_b_i', 'i_vsc_R01_c_r', 'i_vsc_R01_c_i', 'i_vsc_R01_n_r', 'i_vsc_R01_n_i', 'i_vsc_pos_S01_sp', 'i_vsc_S01_sn', 'p_vsc_S01', 'i_vsc_C01_a_r', 'i_vsc_C01_a_i', 'i_vsc_C01_b_r', 'i_vsc_C01_b_i', 'i_vsc_C01_c_r', 'i_vsc_C01_c_i', 'i_vsc_C01_n_r', 'i_vsc_C01_n_i', 'i_vsc_pos_D01_sp', 'i_vsc_D01_sn', 'p_vsc_D01', 'i_vsc_R10_a_r', 'i_vsc_R10_a_i', 'i_vsc_R10_b_r', 'i_vsc_R10_b_i', 'i_vsc_R10_c_r', 'i_vsc_R10_c_i', 'i_vsc_R10_n_r', 'i_vsc_R10_n_i', 'i_vsc_pos_S10_sp', 'i_vsc_S10_sn', 'p_vsc_S10', 'i_vsc_R14_a_r', 'i_vsc_R14_a_i', 'i_vsc_R14_b_r', 'i_vsc_R14_b_i', 'i_vsc_R14_c_r', 'i_vsc_R14_c_i', 'i_vsc_R14_n_r', 'i_vsc_R14_n_i', 'i_vsc_pos_S14_sp', 'i_vsc_S14_sn', 'p_vsc_S14', 'i_vsc_I02_a_r', 'i_vsc_I02_a_i', 'i_vsc_I02_b_r', 'i_vsc_I02_b_i', 'i_vsc_I02_c_r', 'i_vsc_I02_c_i', 'i_vsc_I02_n_r', 'i_vsc_I02_n_i', 'i_vsc_pos_H02_sp', 'i_vsc_H02_sn', 'p_vsc_H02', 'i_vsc_C09_a_r', 'i_vsc_C09_a_i', 'i_vsc_C09_b_r', 'i_vsc_C09_b_i', 'i_vsc_C09_c_r', 'i_vsc_C09_c_i', 'i_vsc_C09_n_r', 'i_vsc_C09_n_i', 'i_vsc_pos_D09_sp', 'i_vsc_D09_sn', 'p_vsc_D09', 'i_vsc_C11_a_r', 'i_vsc_C11_a_i', 'i_vsc_C11_b_r', 'i_vsc_C11_b_i', 'i_vsc_C11_c_r', 'i_vsc_C11_c_i', 'i_vsc_C11_n_r', 'i_vsc_C11_n_i', 'i_vsc_pos_D11_sp', 'i_vsc_D11_sn', 'p_vsc_D11', 'i_vsc_C16_a_r', 'i_vsc_C16_a_i', 'i_vsc_C16_b_r', 'i_vsc_C16_b_i', 'i_vsc_C16_c_r', 'i_vsc_C16_c_i', 'i_vsc_C16_n_r', 'i_vsc_C16_n_i', 'i_vsc_pos_D16_sp', 'i_vsc_D16_sn', 'p_vsc_D16', 'i_vsc_MV0_a_r', 'i_vsc_MV0_b_r', 'i_vsc_MV0_c_r', 'i_vsc_MV0_a_i', 'i_vsc_MV0_b_i', 'i_vsc_MV0_c_i', 'omega_coi', 'p_agc'] 
        self.xy_ini_list = self.x_list + self.y_ini_list 
        self.t = 0.0
        self.it = 0
        self.it_store = 0
        self.xy_prev = np.zeros((self.N_x+self.N_y,1))
        self.initialization_tol = 1e-6
        self.N_u = len(self.inputs_run_list) 
        self.sopt_root_method='hybr'
        self.sopt_root_jac=True
        self.u_ini_list = self.inputs_ini_list
        self.u_ini_values_list = self.inputs_ini_values_list
        self.u_run_list = self.inputs_run_list
        self.u_run_values_list = self.inputs_run_values_list
        self.N_u = len(self.u_run_list)
        self.u_ini = np.array(self.inputs_ini_values_list,dtype=np.float64)
        self.p = np.array(self.params_values_list,dtype=np.float64)
        self.xy_0 = np.zeros((self.N_x+self.N_y,),dtype=np.float64)
        self.xy = np.zeros((self.N_x+self.N_y,),dtype=np.float64)
        self.z = np.zeros((self.N_z,),dtype=np.float64)
        
        # numerical elements of jacobians computing:
        x = self.xy[:self.N_x]
        y = self.xy[self.N_x:]
        
        self.yini2urun = list(set(self.u_run_list).intersection(set(self.y_ini_list)))
        self.uini2yrun = list(set(self.y_run_list).intersection(set(self.u_ini_list)))
        self.Time = np.zeros(self.N_store)
        self.X = np.zeros((self.N_store,self.N_x))
        self.Y = np.zeros((self.N_store,self.N_y))
        self.Z = np.zeros((self.N_store,self.N_z))
        self.iters = np.zeros(self.N_store) 
        self.u_run = np.array(self.u_run_values_list,dtype=np.float64)
 
        ## jac_ini
        self.jac_ini = np.zeros((self.N_x+self.N_y,self.N_x+self.N_y))
        self.sp_jac_ini_ia, self.sp_jac_ini_ja, self.sp_jac_ini_nia, self.sp_jac_ini_nja = sp_jac_ini_vectors()
        data = np.array(self.sp_jac_ini_ia,dtype=np.float64)
        #self.sp_jac_ini = sspa.csr_matrix((data, self.sp_jac_ini_ia, self.sp_jac_ini_ja), shape=(self.sp_jac_ini_nia,self.sp_jac_ini_nja))
           
        if self.dae_file_mode == 'enviroment':
            fobj = BytesIO(pkgutil.get_data(__name__, f'./cigre_eu_lv_acdc_c_sp_jac_ini_num.npz'))
            self.sp_jac_ini = sspa.load_npz(fobj)
        else:
            self.sp_jac_ini = sspa.load_npz(f'./{self.matrices_folder}/cigre_eu_lv_acdc_c_sp_jac_ini_num.npz')
            
            
        self.jac_ini = self.sp_jac_ini.toarray()

        #self.J_ini_d = np.array(self.sp_jac_ini_ia)*0.0
        #self.J_ini_i = np.array(self.sp_jac_ini_ia)
        #self.J_ini_p = np.array(self.sp_jac_ini_ja)
        de_jac_ini_eval(self.jac_ini,x,y,self.u_ini,self.p,self.Dt)
        sp_jac_ini_eval(self.sp_jac_ini.data,x,y,self.u_ini,self.p,self.Dt) 
        self.fill_factor_ini,self.drop_tol_ini,self.drop_rule_ini = 100,1e-10,'basic'       


        ## jac_run
        self.jac_run = np.zeros((self.N_x+self.N_y,self.N_x+self.N_y))
        self.sp_jac_run_ia, self.sp_jac_run_ja, self.sp_jac_run_nia, self.sp_jac_run_nja = sp_jac_run_vectors()
        data = np.array(self.sp_jac_run_ia,dtype=np.float64)

        if self.dae_file_mode == 'enviroment':
            fobj = BytesIO(pkgutil.get_data(__name__, './cigre_eu_lv_acdc_c_sp_jac_run_num.npz'))
            self.sp_jac_run = sspa.load_npz(fobj)
        else:
            self.sp_jac_run = sspa.load_npz(f'./{self.matrices_folder}/cigre_eu_lv_acdc_c_sp_jac_run_num.npz')
        self.jac_run = self.sp_jac_run.toarray()            
           
        self.J_run_d = np.array(self.sp_jac_run_ia)*0.0
        self.J_run_i = np.array(self.sp_jac_run_ia)
        self.J_run_p = np.array(self.sp_jac_run_ja)
        de_jac_run_eval(self.jac_run,x,y,self.u_run,self.p,self.Dt)
        sp_jac_run_eval(self.J_run_d,x,y,self.u_run,self.p,self.Dt)
        
        ## jac_trap
        self.jac_trap = np.zeros((self.N_x+self.N_y,self.N_x+self.N_y))
        self.sp_jac_trap_ia, self.sp_jac_trap_ja, self.sp_jac_trap_nia, self.sp_jac_trap_nja = sp_jac_trap_vectors()
        data = np.array(self.sp_jac_trap_ia,dtype=np.float64)
        #self.sp_jac_trap = sspa.csr_matrix((data, self.sp_jac_trap_ia, self.sp_jac_trap_ja), shape=(self.sp_jac_trap_nia,self.sp_jac_trap_nja))
       
    

        if self.dae_file_mode == 'enviroment':
            fobj = BytesIO(pkgutil.get_data(__name__, './cigre_eu_lv_acdc_c_sp_jac_trap_num.npz'))
            self.sp_jac_trap = sspa.load_npz(fobj)
        else:
            self.sp_jac_trap = sspa.load_npz(f'./{self.matrices_folder}/cigre_eu_lv_acdc_c_sp_jac_trap_num.npz')
            

        self.jac_trap = self.sp_jac_trap.toarray()
        
        #self.J_trap_d = np.array(self.sp_jac_trap_ia)*0.0
        #self.J_trap_i = np.array(self.sp_jac_trap_ia)
        #self.J_trap_p = np.array(self.sp_jac_trap_ja)
        de_jac_trap_eval(self.jac_trap,x,y,self.u_run,self.p,self.Dt)
        sp_jac_trap_eval(self.sp_jac_trap.data,x,y,self.u_run,self.p,self.Dt)
        self.fill_factor_trap,self.drop_tol_trap,self.drop_rule_trap = 100,1e-10,'basic' 
   

        

        
        self.max_it,self.itol,self.store = 50,1e-8,1 
        self.lmax_it,self.ltol,self.ldamp= 50,1e-8,1.0
        self.mode = 0 

        self.lmax_it_ini,self.ltol_ini,self.ldamp_ini=50,1e-8,1.0

        self.sp_Fu_run = sspa.load_npz(f'./{self.matrices_folder}/cigre_eu_lv_acdc_c_Fu_run_num.npz')
        self.sp_Gu_run = sspa.load_npz(f'./{self.matrices_folder}/cigre_eu_lv_acdc_c_Gu_run_num.npz')
        self.sp_Hx_run = sspa.load_npz(f'./{self.matrices_folder}/cigre_eu_lv_acdc_c_Hx_run_num.npz')
        self.sp_Hy_run = sspa.load_npz(f'./{self.matrices_folder}/cigre_eu_lv_acdc_c_Hy_run_num.npz')
        self.sp_Hu_run = sspa.load_npz(f'./{self.matrices_folder}/cigre_eu_lv_acdc_c_Hu_run_num.npz')        
        
        self.ss_solver = 2
        self.lsolver = 2
 
        



        
    def update(self):

        self.Time = np.zeros(self.N_store)
        self.X = np.zeros((self.N_store,self.N_x))
        self.Y = np.zeros((self.N_store,self.N_y))
        self.Z = np.zeros((self.N_store,self.N_z))
        self.iters = np.zeros(self.N_store)
        
    def ss_ini(self):

        xy_ini,it = sstate(self.xy_0,self.u_ini,self.p,self.jac_ini,self.N_x,self.N_y)
        self.xy_ini = xy_ini
        self.N_iters = it
        
        return xy_ini
    
    # def ini(self,up_dict,xy_0={}):

    #     for item in up_dict:
    #         self.set_value(item,up_dict[item])
            
    #     self.xy_ini = self.ss_ini()
    #     self.ini2run()
    #     jac_run_ss_eval_xy(self.jac_run,self.x,self.y_run,self.u_run,self.p)
    #     jac_run_ss_eval_up(self.jac_run,self.x,self.y_run,self.u_run,self.p)
        
    def jac_run_eval(self):
        de_jac_run_eval(self.jac_run,self.x,self.y_run,self.u_run,self.p,self.Dt)
      
    
    def run(self,t_end,up_dict):
        for item in up_dict:
            self.set_value(item,up_dict[item])
            
        t = self.t
        p = self.p
        it = self.it
        it_store = self.it_store
        xy = self.xy
        u = self.u_run
        z = self.z
        
        t,it,it_store,xy = daesolver(t,t_end,it,it_store,xy,u,p,z,
                                  self.jac_trap,
                                  self.Time,
                                  self.X,
                                  self.Y,
                                  self.Z,
                                  self.iters,
                                  self.Dt,
                                  self.N_x,
                                  self.N_y,
                                  self.N_z,
                                  self.decimation,
                                  max_it=self.max_it,itol=self.itol,store=self.store)
        
        self.t = t
        self.it = it
        self.it_store = it_store
        self.xy = xy
        self.z = z
 
    def runsp(self,t_end,up_dict):
        for item in up_dict:
            self.set_value(item,up_dict[item])
            
        t = self.t
        p = self.p
        it = self.it
        it_store = self.it_store
        xy = self.xy
        u = self.u_run
        
        t,it,it_store,xy = daesolver_sp(t,t_end,it,it_store,xy,u,p,
                                  self.sp_jac_trap,
                                  self.Time,
                                  self.X,
                                  self.Y,
                                  self.Z,
                                  self.iters,
                                  self.Dt,
                                  self.N_x,
                                  self.N_y,
                                  self.N_z,
                                  self.decimation,
                                  max_it=50,itol=1e-8,store=1)
        
        self.t = t
        self.it = it
        self.it_store = it_store
        self.xy = xy
        
    def post(self):
        
        self.Time = self.Time[:self.it_store]
        self.X = self.X[:self.it_store]
        self.Y = self.Y[:self.it_store]
        self.Z = self.Z[:self.it_store]
        
    def ini2run(self):
        
        ## y_ini to y_run
        self.y_ini = self.xy_ini[self.N_x:]
        self.y_run = np.copy(self.y_ini)
        self.u_run = np.copy(self.u_ini)
        
        ## y_ini to u_run
        for item in self.yini2urun:
            self.u_run[self.u_run_list.index(item)] = self.y_ini[self.y_ini_list.index(item)]
                
        ## u_ini to y_run
        for item in self.uini2yrun:
            self.y_run[self.y_run_list.index(item)] = self.u_ini[self.u_ini_list.index(item)]
            
        
        self.x = self.xy_ini[:self.N_x]
        self.xy[:self.N_x] = self.x
        self.xy[self.N_x:] = self.y_run
        c_h_eval(self.z,self.x,self.y_run,self.u_ini,self.p,self.Dt)
        

        
    def get_value(self,name):
        
        if name in self.inputs_run_list:
            value = self.u_run[self.inputs_run_list.index(name)]
            return value
            
        if name in self.x_list:
            idx = self.x_list.index(name)
            value = self.xy[idx]
            return value
            
        if name in self.y_run_list:
            idy = self.y_run_list.index(name)
            value = self.xy[self.N_x+idy]
            return value
        
        if name in self.params_list:
            idp = self.params_list.index(name)
            value = self.p[idp]
            return value
            
        if name in self.outputs_list:
            idz = self.outputs_list.index(name)
            value = self.z[idz]
            return value

    def get_values(self,name):
        if name in self.x_list:
            values = self.X[:,self.x_list.index(name)]
        if name in self.y_run_list:
            values = self.Y[:,self.y_run_list.index(name)]
        if name in self.outputs_list:
            values = self.Z[:,self.outputs_list.index(name)]
                        
        return values

    def get_mvalue(self,names):
        '''

        Parameters
        ----------
        names : list
            list of variables names to return each value.

        Returns
        -------
        mvalue : TYPE
            list of value of each variable.

        '''
        mvalue = []
        for name in names:
            mvalue += [self.get_value(name)]
                        
        return mvalue
    
    def set_value(self,name_,value):
        if name_ in self.inputs_ini_list or name_ in self.inputs_run_list:
            if name_ in self.inputs_ini_list:
                self.u_ini[self.inputs_ini_list.index(name_)] = value
            if name_ in self.inputs_run_list:
                self.u_run[self.inputs_run_list.index(name_)] = value
            return
        elif name_ in self.params_list:
            self.p[self.params_list.index(name_)] = value
            return
        else:
            print(f'Input or parameter {name_} not found.')
 
    def report_x(self,value_format='5.2f'):
        for item in self.x_list:
            print(f'{item:5s} = {self.get_value(item):{value_format}}')

    def report_y(self,value_format='5.2f'):
        for item in self.y_run_list:
            print(f'{item:5s} = {self.get_value(item):{value_format}}')
            
    def report_u(self,value_format='5.2f'):
        for item in self.inputs_run_list:
            print(f'{item:5s} ={self.get_value(item):{value_format}}')

    def report_z(self,value_format='5.2f'):
        for item in self.outputs_list:
            print(f'{item:5s} = {self.get_value(item):{value_format}}')

    def report_params(self,value_format='5.2f'):
        for item in self.params_list:
            print(f'{item:5s} ={self.get_value(item):{value_format}}')
            
    def ini(self,up_dict,xy_0={}):
        '''
        Find the steady state of the initialization problem:
            
               0 = f(x,y,u,p) 
               0 = g(x,y,u,p) 

        Parameters
        ----------
        up_dict : dict
            dictionary with all the parameters p and inputs u new values.
        xy_0: if scalar, all the x and y values initial guess are set to the scalar.
              if dict, the initial guesses are applied for the x and y that are in the dictionary
              if string, the initial guess considers a json file with the x and y names and their initial values

        Returns
        -------
        mvalue : TYPE
            list of value of each variable.

        '''
        
        self.it = 0
        self.it_store = 0
        self.t = 0.0
    
        for item in up_dict:
            self.set_value(item,up_dict[item])
            
        if type(xy_0) == dict:
            xy_0_dict = xy_0
            self.dict2xy0(xy_0_dict)
            
        if type(xy_0) == str:
            if xy_0 == 'eval':
                N_x = self.N_x
                self.xy_0_new = np.copy(self.xy_0)*0
                xy0_eval(self.xy_0_new[:N_x],self.xy_0_new[N_x:],self.u_ini,self.p)
                self.xy_0_evaluated = np.copy(self.xy_0_new)
                self.xy_0 = np.copy(self.xy_0_new)
            else:
                self.load_xy_0(file_name = xy_0)
                
        if type(xy_0) == float or type(xy_0) == int:
            self.xy_0 = np.ones(self.N_x+self.N_y,dtype=np.float64)*xy_0

        xy_ini,it = sstate(self.xy_0,self.u_ini,self.p,
                           self.jac_ini,
                           self.N_x,self.N_y,
                           max_it=self.max_it,tol=self.itol)
        
        if it < self.max_it-1:
            
            self.xy_ini = xy_ini
            self.N_iters = it

            self.ini2run()
            
            self.ini_convergence = True
            
        if it >= self.max_it-1:
            print(f'Maximum number of iterations (max_it = {self.max_it}) reached without convergence.')
            self.ini_convergence = False
            
        return self.ini_convergence
            
        


    
    def dict2xy0(self,xy_0_dict):
    
        for item in xy_0_dict:
            if item in self.x_list:
                self.xy_0[self.x_list.index(item)] = xy_0_dict[item]
            if item in self.y_ini_list:
                self.xy_0[self.y_ini_list.index(item) + self.N_x] = xy_0_dict[item]
        
    
    def save_xy_0(self,file_name = 'xy_0.json'):
        xy_0_dict = {}
        for item in self.x_list:
            xy_0_dict.update({item:self.get_value(item)})
        for item in self.y_ini_list:
            xy_0_dict.update({item:self.get_value(item)})
    
        xy_0_str = json.dumps(xy_0_dict, indent=4)
        with open(file_name,'w') as fobj:
            fobj.write(xy_0_str)
    
    def load_xy_0(self,file_name = 'xy_0.json'):
        with open(file_name) as fobj:
            xy_0_str = fobj.read()
        xy_0_dict = json.loads(xy_0_str)
    
        for item in xy_0_dict:
            if item in self.x_list:
                self.xy_0[self.x_list.index(item)] = xy_0_dict[item]
            if item in self.y_ini_list:
                self.xy_0[self.y_ini_list.index(item)+self.N_x] = xy_0_dict[item]            

    def load_params(self,data_input):
    
        if type(data_input) == str:
            json_file = data_input
            self.json_file = json_file
            self.json_data = open(json_file).read().replace("'",'"')
            data = json.loads(self.json_data)
        elif type(data_input) == dict:
            data = data_input
    
        self.data = data
        for item in self.data:
            self.set_value(item, self.data[item])

    def save_params(self,file_name = 'parameters.json'):
        params_dict = {}
        for item in self.params_list:
            params_dict.update({item:self.get_value(item)})

        params_dict_str = json.dumps(params_dict, indent=4)
        with open(file_name,'w') as fobj:
            fobj.write(params_dict_str)

    def save_inputs_ini(self,file_name = 'inputs_ini.json'):
        inputs_ini_dict = {}
        for item in self.inputs_ini_list:
            inputs_ini_dict.update({item:self.get_value(item)})

        inputs_ini_dict_str = json.dumps(inputs_ini_dict, indent=4)
        with open(file_name,'w') as fobj:
            fobj.write(inputs_ini_dict_str)

    def eval_preconditioner_ini(self):
    
        sp_jac_ini_eval(self.sp_jac_ini.data,self.x,self.y_run,self.u_run,self.p,self.Dt)
    
        csc_sp_jac_ini = sspa.csc_matrix(self.sp_jac_ini)
        P_slu = spilu(csc_sp_jac_ini,
                  fill_factor=self.fill_factor_ini,
                  drop_tol=self.drop_tol_ini,
                  drop_rule = self.drop_rule_ini)
    
        self.P_slu = P_slu
        P_d,P_i,P_p,perm_r,perm_c = slu2pydae(P_slu)   
        self.P_d = P_d
        self.P_i = P_i
        self.P_p = P_p
    
        self.perm_r = perm_r
        self.perm_c = perm_c
            
    
    def eval_preconditioner_trap(self):
    
        sp_jac_trap_eval(self.sp_jac_trap.data,self.x,self.y_run,self.u_run,self.p,self.Dt)
    
        #self.sp_jac_trap.data = self.J_trap_d 
        
        csc_sp_jac_trap = sspa.csc_matrix(self.sp_jac_trap)


        P_slu_trap = spilu(csc_sp_jac_trap,
                          fill_factor=self.fill_factor_trap,
                          drop_tol=self.drop_tol_trap,
                          drop_rule = self.drop_rule_trap)
    
        self.P_slu_trap = P_slu_trap
        P_d,P_i,P_p,perm_r,perm_c = slu2pydae(P_slu_trap)   
        self.P_trap_d = P_d
        self.P_trap_i = P_i
        self.P_trap_p = P_p
    
        self.perm_trap_r = perm_r
        self.perm_trap_c = perm_c
        
    def sprun(self,t_end,up_dict):
        
        for item in up_dict:
            self.set_value(item,up_dict[item])
    
        t = self.t
        p = self.p
        it = self.it
        it_store = self.it_store
        xy = self.xy
        u = self.u_run
        z = self.z
        self.iparams_run = np.zeros(10,dtype=np.float64)
    
        t,it,it_store,xy = spdaesolver(t,t_end,it,it_store,xy,u,p,z,
                                  self.sp_jac_trap.data,self.sp_jac_trap.indices,self.sp_jac_trap.indptr,
                                  self.P_trap_d,self.P_trap_i,self.P_trap_p,self.perm_trap_r,self.perm_trap_c,
                                  self.Time,
                                  self.X,
                                  self.Y,
                                  self.Z,
                                  self.iters,
                                  self.Dt,
                                  self.N_x,
                                  self.N_y,
                                  self.N_z,
                                  self.decimation,
                                  self.iparams_run,
                                  max_it=self.max_it,itol=self.max_it,store=self.store,
                                  lmax_it=self.lmax_it,ltol=self.ltol,ldamp=self.ldamp,mode=self.mode,
                                  lsolver = self.lsolver)
    
        self.t = t
        self.it = it
        self.it_store = it_store
        self.xy = xy
        self.z = z

            
    def spini(self,up_dict,xy_0={}):
    
        self.it = 0
        self.it_store = 0
        self.t = 0.0
    
        for item in up_dict:
            self.set_value(item,up_dict[item])
    
        if type(xy_0) == dict:
            xy_0_dict = xy_0
            self.dict2xy0(xy_0_dict)
    
        if type(xy_0) == str:
            if xy_0 == 'eval':
                N_x = self.N_x
                self.xy_0_new = np.copy(self.xy_0)*0
                xy0_eval(self.xy_0_new[:N_x],self.xy_0_new[N_x:],self.u_ini,self.p)
                self.xy_0_evaluated = np.copy(self.xy_0_new)
                self.xy_0 = np.copy(self.xy_0_new)
            else:
                self.load_xy_0(file_name = xy_0)

        self.xy_ini = self.spss_ini()


        if self.N_iters < self.max_it:
            
            self.ini2run()           
            self.ini_convergence = True
            
        if self.N_iters >= self.max_it:
            print(f'Maximum number of iterations (max_it = {self.max_it}) reached without convergence.')
            self.ini_convergence = False
            
        #jac_run_eval_xy(self.jac_run,self.x,self.y_run,self.u_run,self.p)
        #jac_run_eval_up(self.jac_run,self.x,self.y_run,self.u_run,self.p)
        
        return self.ini_convergence

        
    def spss_ini(self):
        J_d,J_i,J_p = csr2pydae(self.sp_jac_ini)
        
        xy_ini,it,iparams = spsstate(self.xy,self.u_ini,self.p,
                 self.sp_jac_ini.data,self.sp_jac_ini.indices,self.sp_jac_ini.indptr,
                 self.P_d,self.P_i,self.P_p,self.perm_r,self.perm_c,
                 self.N_x,self.N_y,
                 max_it=self.max_it,tol=self.itol,
                 lmax_it=self.lmax_it_ini,
                 ltol=self.ltol_ini,
                 ldamp=self.ldamp,solver=self.ss_solver)

 
        self.xy_ini = xy_ini
        self.N_iters = it
        self.iparams = iparams
    
        return xy_ini

    #def import_cffi(self):
        

    def eval_jac_u2z(self):

        '''

        0 =   J_run * xy + FG_u * u
        z = Hxy_run * xy + H_u * u

        xy = -1/J_run * FG_u * u
        z = -Hxy_run/J_run * FG_u * u + H_u * u
        z = (-Hxy_run/J_run * FG_u + H_u ) * u 
        '''
        
        sp_Fu_run_eval(self.sp_Fu_run.data,self.x,self.y_run,self.u_run,self.p,self.Dt)
        sp_Gu_run_eval(self.sp_Gu_run.data,self.x,self.y_run,self.u_run,self.p,self.Dt)
        sp_H_jacs_run_eval(self.sp_Hx_run.data,
                        self.sp_Hy_run.data,
                        self.sp_Hu_run.data,
                        self.x,self.y_run,self.u_run,self.p,self.Dt)
        sp_jac_run = self.sp_jac_run
        sp_jac_run_eval(sp_jac_run.data,
                        self.x,self.y_run,
                        self.u_run,self.p,
                        self.Dt)



        Hxy_run = sspa.bmat([[self.sp_Hx_run,self.sp_Hy_run]])
        FGu_run = sspa.bmat([[self.sp_Fu_run],[self.sp_Gu_run]])
        

        #((sspa.linalg.spsolve(s.sp_jac_ini,-Hxy_run)) @ FGu_run + sp_Hu_run )@s.u_ini

        self.jac_u2z = Hxy_run @ sspa.linalg.spsolve(self.sp_jac_run,-FGu_run) + self.sp_Hu_run  
        
        
    def step(self,t_end,up_dict):
        for item in up_dict:
            self.set_value(item,up_dict[item])

        t = self.t
        p = self.p
        it = self.it
        it_store = self.it_store
        xy = self.xy
        u = self.u_run
        z = self.z

        t,it,xy = daestep(t,t_end,it,
                          xy,u,p,z,
                          self.jac_trap,
                          self.iters,
                          self.Dt,
                          self.N_x,
                          self.N_y,
                          self.N_z,
                          max_it=self.max_it,itol=self.itol,store=self.store)

        self.t = t
        self.it = it
        self.it_store = it_store
        self.xy = xy
        self.z = z
           
            
    def save_run(self,file_name):
        np.savez(file_name,Time=self.Time,
             X=self.X,Y=self.Y,Z=self.Z,
             x_list = self.x_list,
             y_ini_list = self.y_ini_list,
             y_run_list = self.y_run_list,
             u_ini_list=self.u_ini_list,
             u_run_list=self.u_run_list,  
             z_list=self.outputs_list, 
            )
        
    def load_run(self,file_name):
        data = np.load(f'{file_name}.npz')
        self.Time = data['Time']
        self.X = data['X']
        self.Y = data['Y']
        self.Z = data['Z']
        self.x_list = list(data['x_list'] )
        self.y_run_list = list(data['y_run_list'] )
        self.outputs_list = list(data['z_list'] )
        
    def full_jacs_eval(self):
        N_x = self.N_x
        N_y = self.N_y
        N_xy = N_x + N_y
    
        sp_jac_run = self.sp_jac_run
        sp_Fu = self.sp_Fu_run
        sp_Gu = self.sp_Gu_run
        sp_Hx = self.sp_Hx_run
        sp_Hy = self.sp_Hy_run
        sp_Hu = self.sp_Hu_run
        
        x = self.xy[0:N_x]
        y = self.xy[N_x:]
        u = self.u_run
        p = self.p
        Dt = self.Dt
    
        sp_jac_run_eval(sp_jac_run.data,x,y,u,p,Dt)
        
        self.Fx = sp_jac_run[0:N_x,0:N_x]
        self.Fy = sp_jac_run[ 0:N_x,N_x:]
        self.Gx = sp_jac_run[ N_x:,0:N_x]
        self.Gy = sp_jac_run[ N_x:, N_x:]
        
        sp_Fu_run_eval(sp_Fu.data,x,y,u,p,Dt)
        sp_Gu_run_eval(sp_Gu.data,x,y,u,p,Dt)
        sp_H_jacs_run_eval(sp_Hx.data,sp_Hy.data,sp_Hu.data,x,y,u,p,Dt)
        
        self.Fu = sp_Fu
        self.Gu = sp_Gu
        self.Hx = sp_Hx
        self.Hy = sp_Hy
        self.Hu = sp_Hu


@numba.njit() 
def daestep(t,t_end,it,xy,u,p,z,jac_trap,iters,Dt,N_x,N_y,N_z,max_it=50,itol=1e-8,store=1): 


    fg = np.zeros((N_x+N_y,1),dtype=np.float64)
    fg_i = np.zeros((N_x+N_y),dtype=np.float64)
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    #h = np.zeros((N_z),dtype=np.float64)
    
    f_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    z_ptr=ffi.from_buffer(np.ascontiguousarray(z))
    x_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    jac_trap_ptr=ffi.from_buffer(np.ascontiguousarray(jac_trap))
    
    #de_jac_trap_num_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)    
    de_jac_trap_up_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    de_jac_trap_xy_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    
    if it == 0:
        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        h_eval(z_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        it_store = 0  

    while t<t_end: 
        it += 1
        t += Dt

        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)

        x_0 = np.copy(x) 
        y_0 = np.copy(y) 
        f_0 = np.copy(f) 
        g_0 = np.copy(g) 
            
        for iti in range(max_it):
            f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
            g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
            de_jac_trap_xy_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 

            f_n_i = x - x_0 - 0.5*Dt*(f+f_0) 

            fg_i[:N_x] = f_n_i
            fg_i[N_x:] = g
            
            Dxy_i = np.linalg.solve(-jac_trap,fg_i) 

            x += Dxy_i[:N_x]
            y += Dxy_i[N_x:] 
            
            #print(Dxy_i)

            # iteration stop
            max_relative = 0.0
            for it_var in range(N_x+N_y):
                abs_value = np.abs(xy[it_var])
                if abs_value < 0.001:
                    abs_value = 0.001
                relative_error = np.abs(Dxy_i[it_var])/abs_value

                if relative_error > max_relative: max_relative = relative_error

            if max_relative<itol:
                break
                
        h_eval(z_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        xy[:N_x] = x
        xy[N_x:] = y
        
    return t,it,xy


def daesolver_sp(t,t_end,it,it_store,xy,u,p,sp_jac_trap,T,X,Y,Z,iters,Dt,N_x,N_y,N_z,decimation,max_it=50,itol=1e-8,store=1): 

    fg = np.zeros((N_x+N_y,1),dtype=np.float64)
    fg_i = np.zeros((N_x+N_y),dtype=np.float64)
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    h = np.zeros((N_z),dtype=np.float64)
    sp_jac_trap_eval_up(sp_jac_trap.data,x,y,u,p,Dt,xyup=1)
    
    if it == 0:
        f_run_eval(f,x,y,u,p)
        h_eval(h,x,y,u,p)
        it_store = 0  
        T[0] = t 
        X[0,:] = x  
        Y[0,:] = y  
        Z[0,:] = h  

    while t<t_end: 
        it += 1
        t += Dt

        f_run_eval(f,x,y,u,p)
        g_run_eval(g,x,y,u,p)

        x_0 = np.copy(x) 
        y_0 = np.copy(y) 
        f_0 = np.copy(f) 
        g_0 = np.copy(g) 
            
        for iti in range(max_it):
            f_run_eval(f,x,y,u,p)
            g_run_eval(g,x,y,u,p)
            sp_jac_trap_eval(sp_jac_trap.data,x,y,u,p,Dt,xyup=1)            

            f_n_i = x - x_0 - 0.5*Dt*(f+f_0) 

            fg_i[:N_x] = f_n_i
            fg_i[N_x:] = g
            
            Dxy_i = spsolve(sp_jac_trap,-fg_i) 

            x = x + Dxy_i[:N_x]
            y = y + Dxy_i[N_x:]              

            # iteration stop
            max_relative = 0.0
            for it_var in range(N_x+N_y):
                abs_value = np.abs(xy[it_var])
                if abs_value < 0.001:
                    abs_value = 0.001
                relative_error = np.abs(Dxy_i[it_var])/abs_value

                if relative_error > max_relative: max_relative = relative_error

            if max_relative<itol:
                break
                
        h_eval(h,x,y,u,p)
        xy[:N_x] = x
        xy[N_x:] = y
        
        # store in channels 
        if store == 1:
            if it >= it_store*decimation: 
                T[it_store+1] = t 
                X[it_store+1,:] = x 
                Y[it_store+1,:] = y
                Z[it_store+1,:] = h
                iters[it_store+1] = iti
                it_store += 1 

    return t,it,it_store,xy




@numba.njit()
def sprichardson(A_d,A_i,A_p,b,P_d,P_i,P_p,perm_r,perm_c,x,iparams,damp=1.0,max_it=100,tol=1e-3):
    N_A = A_p.shape[0]-1
    f = np.zeros(N_A)
    for it in range(max_it):
        spMvmul(N_A,A_d,A_i,A_p,x,f) 
        f -= b                          # A@x-b
        x = x - damp*splu_solve(P_d,P_i,P_p,perm_r,perm_c,f)   
        if np.linalg.norm(f,2) < tol: break
    iparams[0] = it
    return x
    
@numba.njit()
def spconjgradm(A_d,A_i,A_p,b,P_d,P_i,P_p,perm_r,perm_c,x,iparams,max_it=100,tol=1e-3, damp=None):
    """
    A function to solve [A]{x} = {b} linear equation system with the 
    preconditioned conjugate gradient method.
    More at: http://en.wikipedia.org/wiki/Conjugate_gradient_method
    ========== Parameters ==========
    A_d,A_i,A_p : sparse matrix 
        components in CRS form A_d = A_crs.data, A_i = A_crs.indices, A_p = A_crs.indptr.
    b : vector
        The right hand side (RHS) vector of the system.
    x : vector
        The starting guess for the solution.
    P_d,P_i,P_p,perm_r,perm_c: preconditioner LU matrix
        components in scipy.spilu form P_d,P_i,P_p,perm_r,perm_c = slu2pydae(M)
        with M = scipy.sparse.linalg.spilu(A_csc) 

    """  
    N   = len(b)
    Ax  = np.zeros(N)
    Ap  = np.zeros(N)
    App = np.zeros(N)
    pAp = np.zeros(N)
    z   = np.zeros(N)
    
    spMvmul(N,A_d,A_i,A_p,x,Ax)
    r = -(Ax - b)
    z = splu_solve(P_d,P_i,P_p,perm_r,perm_c,r) #z = M.solve(r)
    p = z
    zsold = 0.0
    for it in range(N):  # zsold = np.dot(np.transpose(z), z)
        zsold += z[it]*z[it]
    for i in range(max_it):
        spMvmul(N,A_d,A_i,A_p,p,App)  # #App = np.dot(A, p)
        Ap = splu_solve(P_d,P_i,P_p,perm_r,perm_c,App) #Ap = M.solve(App)
        pAp = 0.0
        for it in range(N):
            pAp += p[it]*Ap[it]

        alpha = zsold / pAp
        x = x + alpha*p
        z = z - alpha*Ap
        zz = 0.0
        for it in range(N):  # z.T@z
            zz += z[it]*z[it]
        zsnew = zz
        if np.sqrt(zsnew) < tol:
            break
            
        p = z + (zsnew/zsold)*p
        zsold = zsnew
    iparams[0] = i

    return x


@numba.njit()
def spsstate(xy,u,p,
             J_d,J_i,J_p,
             P_d,P_i,P_p,perm_r,perm_c,
             N_x,N_y,
             max_it=50,tol=1e-8,
             lmax_it=20,ltol=1e-8,ldamp=1.0, solver=2):
    
   
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    iparams = np.array([0],dtype=np.int64)    
    
    f_c_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_c_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))
    J_d_ptr=ffi.from_buffer(np.ascontiguousarray(J_d))

    #sp_jac_ini_num_eval(J_d_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
    sp_jac_ini_up_eval(J_d_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
    
    #sp_jac_ini_eval_up(J_d,x,y,u,p,0.0)

    Dxy = np.zeros(N_x + N_y)
    for it in range(max_it):
        
        x = xy[:N_x]
        y = xy[N_x:]   
       
        sp_jac_ini_xy_eval(J_d_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)

        
        f_ini_eval(f_c_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
        g_ini_eval(g_c_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
        
        #f_ini_eval(f,x,y,u,p)
        #g_ini_eval(g,x,y,u,p)
        
        fg[:N_x] = f
        fg[N_x:] = g
        
        if solver==1:
               
            Dxy = sprichardson(J_d,J_i,J_p,-fg,P_d,P_i,P_p,perm_r,perm_c,Dxy,iparams,damp=ldamp,max_it=lmax_it,tol=ltol)
   
        if solver==2:
            
            Dxy = spconjgradm(J_d,J_i,J_p,-fg,P_d,P_i,P_p,perm_r,perm_c,Dxy,iparams,damp=ldamp,max_it=lmax_it,tol=ltol)
            
        xy += Dxy
        #if np.max(np.abs(fg))<tol: break
        if np.linalg.norm(fg,np.inf)<tol: break

    return xy,it,iparams


    
@numba.njit() 
def daesolver(t,t_end,it,it_store,xy,u,p,z,jac_trap,T,X,Y,Z,iters,Dt,N_x,N_y,N_z,decimation,max_it=50,itol=1e-8,store=1): 


    fg = np.zeros((N_x+N_y,1),dtype=np.float64)
    fg_i = np.zeros((N_x+N_y),dtype=np.float64)
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    #h = np.zeros((N_z),dtype=np.float64)
    
    f_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    z_ptr=ffi.from_buffer(np.ascontiguousarray(z))
    x_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    jac_trap_ptr=ffi.from_buffer(np.ascontiguousarray(jac_trap))
    
    #de_jac_trap_num_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)    
    de_jac_trap_up_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    de_jac_trap_xy_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    
    if it == 0:
        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        h_eval(z_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        it_store = 0  
        T[0] = t 
        X[0,:] = x  
        Y[0,:] = y  
        Z[0,:] = z  

    while t<t_end: 
        it += 1
        t += Dt

        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)

        x_0 = np.copy(x) 
        y_0 = np.copy(y) 
        f_0 = np.copy(f) 
        g_0 = np.copy(g) 
            
        for iti in range(max_it):
            f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
            g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
            de_jac_trap_xy_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 

            f_n_i = x - x_0 - 0.5*Dt*(f+f_0) 

            fg_i[:N_x] = f_n_i
            fg_i[N_x:] = g
            
            Dxy_i = np.linalg.solve(-jac_trap,fg_i) 

            x += Dxy_i[:N_x]
            y += Dxy_i[N_x:] 
            
            #print(Dxy_i)

            # iteration stop
            max_relative = 0.0
            for it_var in range(N_x+N_y):
                abs_value = np.abs(xy[it_var])
                if abs_value < 0.001:
                    abs_value = 0.001
                relative_error = np.abs(Dxy_i[it_var])/abs_value

                if relative_error > max_relative: max_relative = relative_error

            if max_relative<itol:
                break
                
        h_eval(z_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        xy[:N_x] = x
        xy[N_x:] = y
        
        # store in channels 
        if store == 1:
            if it >= it_store*decimation: 
                T[it_store+1] = t 
                X[it_store+1,:] = x 
                Y[it_store+1,:] = y
                Z[it_store+1,:] = z
                iters[it_store+1] = iti
                it_store += 1 

    return t,it,it_store,xy
    
@numba.njit() 
def spdaesolver(t,t_end,it,it_store,xy,u,p,z,
                J_d,J_i,J_p,
                P_d,P_i,P_p,perm_r,perm_c,
                T,X,Y,Z,iters,Dt,N_x,N_y,N_z,decimation,
                iparams,
                max_it=50,itol=1e-8,store=1,
                lmax_it=20,ltol=1e-4,ldamp=1.0,mode=0,lsolver=2):

    fg_i = np.zeros((N_x+N_y),dtype=np.float64)
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    z = np.zeros((N_z),dtype=np.float64)
    Dxy_i_0 = np.zeros(N_x+N_y,dtype=np.float64) 
    f_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    z_ptr=ffi.from_buffer(np.ascontiguousarray(z))
    x_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    J_d_ptr=ffi.from_buffer(np.ascontiguousarray(J_d))
    
    #sp_jac_trap_num_eval(J_d_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)    
    sp_jac_trap_up_eval( J_d_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    sp_jac_trap_xy_eval( J_d_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    
    if it == 0:
        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        h_eval(z_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        it_store = 0  
        T[0] = t 
        X[0,:] = x  
        Y[0,:] = y  
        Z[0,:] = z 

    while t<t_end: 
        it += 1
        t += Dt

        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)

        x_0 = np.copy(x) 
        y_0 = np.copy(y) 
        f_0 = np.copy(f) 
        g_0 = np.copy(g) 
            
        for iti in range(max_it):
            f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
            g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
            sp_jac_trap_xy_eval(J_d_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 

            f_n_i = x - x_0 - 0.5*Dt*(f+f_0) 

            fg_i[:N_x] = f_n_i
            fg_i[N_x:] = g
            
            #Dxy_i = np.linalg.solve(-jac_trap,fg_i) 
            if lsolver == 1:
                Dxy_i = sprichardson(J_d,J_i,J_p,-fg_i,P_d,P_i,P_p,perm_r,perm_c,
                                     Dxy_i_0,iparams,damp=ldamp,max_it=lmax_it,tol=ltol)
            if lsolver == 2:
                Dxy_i = spconjgradm(J_d,J_i,J_p,-fg_i,P_d,P_i,P_p,perm_r,perm_c,
                                     Dxy_i_0,iparams,damp=ldamp,max_it=lmax_it,tol=ltol)                

            x += Dxy_i[:N_x]
            y += Dxy_i[N_x:] 
            
            #print(Dxy_i)

            # iteration stop
            max_relative = 0.0
            for it_var in range(N_x+N_y):
                abs_value = np.abs(xy[it_var])
                if abs_value < 0.001:
                    abs_value = 0.001
                relative_error = np.abs(Dxy_i[it_var])/abs_value

                if relative_error > max_relative: max_relative = relative_error

            if max_relative<itol:
                break
                
        h_eval(z_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        xy[:N_x] = x
        xy[N_x:] = y
        
        # store in channels 
        if store == 1:
            if it >= it_store*decimation: 
                T[it_store+1] = t 
                X[it_store+1,:] = x 
                Y[it_store+1,:] = y
                Z[it_store+1,:] = z
                iters[it_store+1] = iti
                it_store += 1 

    return t,it,it_store,xy


@cuda.jit()
def ode_solve(x,u,p,f_run,u_idxs,z_i,z,sim):

    N_i,N_j,N_x,N_z,Dt = sim

    # index of thread on GPU:
    i = cuda.grid(1)

    if i < x.size:
        for j in range(N_j):
            f_run_eval(f_run[i,:],x[i,:],u[i,u_idxs[j],:],p[i,:])
            for k in range(N_x):
              x[i,k] +=  Dt*f_run[i,k]

            # outputs in time range
            #z[i,j] = u[i,idxs[j],0]
            z[i,j] = x[i,1]
        h_eval(z_i[i,:],x[i,:],u[i,u_idxs[j],:],p[i,:])
        
def csr2pydae(A_csr):
    '''
    From scipy CSR to the three vectors:
    
    - data
    - indices
    - indptr
    
    '''
    
    A_d = A_csr.data
    A_i = A_csr.indices
    A_p = A_csr.indptr
    
    return A_d,A_i,A_p
    
def slu2pydae(P_slu):
    '''
    From SupderLU matrix to the three vectors:
    
    - data
    - indices
    - indptr
    
    and the premutation vectors:
    
    - perm_r
    - perm_c
    
    '''
    N = P_slu.shape[0]
    #P_slu_full = P_slu.L.A - sspa.eye(N,format='csr') + P_slu.U.A
    P_slu_full = P_slu.L - sspa.eye(N,format='csc') + P_slu.U
    perm_r = P_slu.perm_r
    perm_c = P_slu.perm_c
    P_csr = sspa.csr_matrix(P_slu_full)
    
    P_d = P_csr.data
    P_i = P_csr.indices
    P_p = P_csr.indptr
    
    return P_d,P_i,P_p,perm_r,perm_c

@numba.njit(cache=True)
def spMvmul(N,A_data,A_indices,A_indptr,x,y):
    '''
    y = A @ x
    
    with A in sparse CRS form
    '''
    #y = np.zeros(x.shape[0])
    for i in range(N):
        y[i] = 0.0
        for j in range(A_indptr[i],A_indptr[i + 1]):
            y[i] = y[i] + A_data[j]*x[A_indices[j]]
            
            
@numba.njit(cache=True)
def splu_solve(LU_d,LU_i,LU_p,perm_r,perm_c,b):
    N = len(b)
    y = np.zeros(N)
    x = np.zeros(N)
    z = np.zeros(N)
    bp = np.zeros(N)
    
    for i in range(N): 
        bp[perm_r[i]] = b[i]
        
    for i in range(N): 
        y[i] = bp[i]
        for j in range(LU_p[i],LU_p[i+1]):
            if LU_i[j]>i-1: break
            y[i] -= LU_d[j] * y[LU_i[j]]

    for i in range(N-1,-1,-1): #(int i = N - 1; i >= 0; i--) 
        z[i] = y[i]
        den = 0.0
        for j in range(LU_p[i],LU_p[i+1]): #(int k = i + 1; k < N; k++)
            if LU_i[j] > i:
                z[i] -= LU_d[j] * z[LU_i[j]]
            if LU_i[j] == i: den = LU_d[j]
        z[i] = z[i]/den
 
    for i in range(N):
        x[i] = z[perm_c[i]]
        
    return x



@numba.njit("float64[:,:](float64[:,:],float64[:],float64[:],float64[:],float64[:],float64)")
def de_jac_ini_eval(de_jac_ini,x,y,u,p,Dt):   
    '''
    Computes the dense full initialization jacobian:
    
    jac_ini = [[Fx_ini, Fy_ini],
               [Gx_ini, Gy_ini]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_ini : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (ini problem).
    u : (N_u,) array_like
        Vector with inputs (ini problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with N = N_x+N_y
 
    Returns
    -------
    
    de_jac_ini : (N, N) array_like
                  Updated matrix.    
    
    '''
    
    de_jac_ini_ptr=ffi.from_buffer(np.ascontiguousarray(de_jac_ini))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    de_jac_ini_num_eval(de_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    de_jac_ini_up_eval( de_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    de_jac_ini_xy_eval( de_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return de_jac_ini

@numba.njit("float64[:,:](float64[:,:],float64[:],float64[:],float64[:],float64[:],float64)")
def de_jac_run_eval(de_jac_run,x,y,u,p,Dt):   
    '''
    Computes the dense full initialization jacobian:
    
    jac_run = [[Fx_run, Fy_run],
               [Gx_run, Gy_run]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_run : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (ini problem).
    u : (N_u,) array_like
        Vector with inputs (ini problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with N = N_x+N_y
 
    Returns
    -------
    
    de_jac_ini : (N, N) array_like
                  Updated matrix.    
    
    '''
    
    de_jac_run_ptr=ffi.from_buffer(np.ascontiguousarray(de_jac_run))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    de_jac_run_num_eval(de_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    de_jac_run_up_eval( de_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    de_jac_run_xy_eval( de_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return de_jac_run

@numba.njit("float64[:,:](float64[:,:],float64[:],float64[:],float64[:],float64[:],float64)")
def de_jac_trap_eval(de_jac_trap,x,y,u,p,Dt):   
    '''
    Computes the dense full trapezoidal jacobian:
    
    jac_trap = [[eye - 0.5*Dt*Fx_run, -0.5*Dt*Fy_run],
                [             Gx_run,         Gy_run]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_trap : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (run problem).
    u : (N_u,) array_like
        Vector with inputs (run problem). 
    p : (N_p,) array_like
        Vector with parameters. 
 
    Returns
    -------
    
    de_jac_trap : (N, N) array_like
                  Updated matrix.    
    
    '''
        
    de_jac_trap_ptr = ffi.from_buffer(np.ascontiguousarray(de_jac_trap))
    x_c_ptr = ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr = ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr = ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr = ffi.from_buffer(np.ascontiguousarray(p))

    de_jac_trap_num_eval(de_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    de_jac_trap_up_eval( de_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    de_jac_trap_xy_eval( de_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return de_jac_trap


@numba.njit("float64[:](float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def sp_jac_run_eval(sp_jac_run,x,y,u,p,Dt):   
    '''
    Computes the sparse full trapezoidal jacobian:
    
    jac_trap = [[eye - 0.5*Dt*Fx_run, -0.5*Dt*Fy_run],
                [             Gx_run,         Gy_run]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    sp_jac_trap : (Nnz,) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (run problem).
    u : (N_u,) array_like
        Vector with inputs (run problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with Nnz the number of non-zeros elements in the jacobian.
 
    Returns
    -------
    
    sp_jac_trap : (Nnz,) array_like
                  Updated matrix.    
    
    '''        
    sp_jac_run_ptr=ffi.from_buffer(np.ascontiguousarray(sp_jac_run))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    sp_jac_run_num_eval( sp_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_jac_run_up_eval( sp_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_jac_run_xy_eval( sp_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return sp_jac_run

@numba.njit("float64[:](float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def sp_jac_trap_eval(sp_jac_trap,x,y,u,p,Dt):   
    '''
    Computes the sparse full trapezoidal jacobian:
    
    jac_trap = [[eye - 0.5*Dt*Fx_run, -0.5*Dt*Fy_run],
                [             Gx_run,         Gy_run]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    sp_jac_trap : (Nnz,) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (run problem).
    u : (N_u,) array_like
        Vector with inputs (run problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with Nnz the number of non-zeros elements in the jacobian.
 
    Returns
    -------
    
    sp_jac_trap : (Nnz,) array_like
                  Updated matrix.    
    
    '''        
    sp_jac_trap_ptr=ffi.from_buffer(np.ascontiguousarray(sp_jac_trap))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    sp_jac_trap_num_eval(sp_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_jac_trap_up_eval( sp_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_jac_trap_xy_eval( sp_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return sp_jac_trap

@numba.njit("float64[:](float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def sp_jac_ini_eval(sp_jac_ini,x,y,u,p,Dt):   
    '''
    Computes the SPARSE full initialization jacobian:
    
    jac_ini = [[Fx_ini, Fy_ini],
               [Gx_ini, Gy_ini]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_ini : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (ini problem).
    u : (N_u,) array_like
        Vector with inputs (ini problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with N = N_x+N_y
 
    Returns
    -------
    
    de_jac_ini : (N, N) array_like
                  Updated matrix.    
    
    '''
    
    sp_jac_ini_ptr=ffi.from_buffer(np.ascontiguousarray(sp_jac_ini))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    sp_jac_ini_num_eval(sp_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_jac_ini_up_eval( sp_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_jac_ini_xy_eval( sp_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return sp_jac_ini


@numba.njit()
def sstate(xy,u,p,jac_ini_ss,N_x,N_y,max_it=50,tol=1e-8):
    
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]

    f_c_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_c_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))
    jac_ini_ss_ptr=ffi.from_buffer(np.ascontiguousarray(jac_ini_ss))

    #de_jac_ini_num_eval(jac_ini_ss_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
    de_jac_ini_up_eval(jac_ini_ss_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)

    for it in range(max_it):
        de_jac_ini_xy_eval(jac_ini_ss_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
        f_ini_eval(f_c_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
        g_ini_eval(g_c_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
        fg[:N_x] = f
        fg[N_x:] = g
        xy += np.linalg.solve(jac_ini_ss,-fg)
        if np.max(np.abs(fg))<tol: break

    return xy,it


@numba.njit("float64[:](float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def c_h_eval(z,x,y,u,p,Dt):   
    '''
    Computes the SPARSE full initialization jacobian:
    
    jac_ini = [[Fx_ini, Fy_ini],
               [Gx_ini, Gy_ini]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_ini : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (ini problem).
    u : (N_u,) array_like
        Vector with inputs (ini problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with N = N_x+N_y
 
    Returns
    -------
    
    de_jac_ini : (N, N) array_like
                  Updated matrix.    
    
    '''
    
    z_c_ptr=ffi.from_buffer(np.ascontiguousarray(z))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    h_eval(z_c_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return z

@numba.njit("(float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def sp_Fu_run_eval(jac,x,y,u,p,Dt):   
    '''
    Computes the dense full initialization jacobian:
    
    jac_ini = [[Fx_ini, Fy_ini],
               [Gx_ini, Gy_ini]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_ini : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (ini problem).
    u : (N_u,) array_like
        Vector with inputs (ini problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with N = N_x+N_y
 
    Returns
    -------
    
    de_jac_ini : (N, N) array_like
                  Updated matrix.    
    
    '''
    
    jac_ptr=ffi.from_buffer(np.ascontiguousarray(jac))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    sp_Fu_run_up_eval( jac_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Fu_run_xy_eval( jac_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    #return jac

@numba.njit("(float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def sp_Gu_run_eval(jac,x,y,u,p,Dt):   
    '''
    Computes the dense full initialization jacobian:
    
    jac_ini = [[Fx_ini, Fy_ini],
               [Gx_ini, Gy_ini]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_ini : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (ini problem).
    u : (N_u,) array_like
        Vector with inputs (ini problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with N = N_x+N_y
 
    Returns
    -------
    
    de_jac_ini : (N, N) array_like
                  Updated matrix.    
    
    '''
    
    jac_ptr=ffi.from_buffer(np.ascontiguousarray(jac))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    sp_Gu_run_up_eval( jac_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Gu_run_xy_eval( jac_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    #return jac

@numba.njit("(float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def sp_H_jacs_run_eval(H_x,H_y,H_u,x,y,u,p,Dt):   
    '''
    Computes the dense full initialization jacobian:
    
    jac_ini = [[Fx_ini, Fy_ini],
               [Gx_ini, Gy_ini]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_ini : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (ini problem).
    u : (N_u,) array_like
        Vector with inputs (ini problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with N = N_x+N_y
 
    Returns
    -------
    
    de_jac_ini : (N, N) array_like
                  Updated matrix.    
    
    '''
    
    H_x_ptr=ffi.from_buffer(np.ascontiguousarray(H_x))
    H_y_ptr=ffi.from_buffer(np.ascontiguousarray(H_y))
    H_u_ptr=ffi.from_buffer(np.ascontiguousarray(H_u))

    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    sp_Hx_run_up_eval( H_x_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Hx_run_xy_eval( H_x_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Hy_run_up_eval( H_y_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Hy_run_xy_eval( H_y_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Hu_run_up_eval( H_u_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Hu_run_xy_eval( H_u_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)





def sp_jac_ini_vectors():

    sp_jac_ini_ia = [0, 695, 1, 31, 32, 37, 38, 351, 352, 353, 354, 2, 33, 34, 37, 38, 351, 352, 353, 354, 3, 35, 36, 37, 38, 351, 352, 353, 354, 4, 103, 104, 109, 110, 375, 376, 377, 378, 5, 105, 106, 109, 110, 375, 376, 377, 378, 6, 107, 108, 109, 110, 375, 376, 377, 378, 7, 135, 136, 141, 142, 395, 396, 397, 398, 8, 137, 138, 141, 142, 395, 396, 397, 398, 9, 139, 140, 141, 142, 395, 396, 397, 398, 10, 183, 184, 189, 190, 407, 408, 409, 410, 11, 185, 186, 189, 190, 407, 408, 409, 410, 12, 187, 188, 189, 190, 407, 408, 409, 410, 13, 191, 192, 197, 198, 411, 412, 413, 414, 14, 193, 194, 197, 198, 411, 412, 413, 414, 15, 195, 196, 197, 198, 411, 412, 413, 414, 16, 255, 256, 261, 262, 427, 428, 429, 430, 17, 257, 258, 261, 262, 427, 428, 429, 430, 18, 259, 260, 261, 262, 427, 428, 429, 430, 19, 271, 272, 277, 278, 431, 432, 433, 434, 20, 273, 274, 277, 278, 431, 432, 433, 434, 21, 275, 276, 277, 278, 431, 432, 433, 434, 22, 311, 312, 317, 318, 439, 440, 441, 442, 23, 313, 314, 317, 318, 439, 440, 441, 442, 24, 315, 316, 317, 318, 439, 440, 441, 442, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 175, 176, 177, 178, 191, 192, 193, 194, 689, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 175, 176, 177, 178, 191, 192, 193, 194, 692, 25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 177, 178, 179, 180, 193, 194, 195, 196, 690, 25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 177, 178, 179, 180, 193, 194, 195, 196, 693, 25, 26, 27, 28, 29, 30, 31, 32, 35, 36, 175, 176, 179, 180, 191, 192, 195, 196, 691, 25, 26, 27, 28, 29, 30, 31, 32, 35, 36, 175, 176, 179, 180, 191, 192, 195, 196, 694, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 455, 601, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 456, 602, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 457, 603, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 458, 604, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 459, 605, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 460, 606, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 461, 607, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 462, 608, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117, 118, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117, 118, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117, 118, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117, 118, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117, 118, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117, 118, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117, 118, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117, 118, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 119, 120, 121, 122, 123, 124, 125, 126, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 119, 120, 121, 122, 123, 124, 125, 126, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 119, 120, 121, 122, 123, 124, 125, 126, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 119, 120, 121, 122, 123, 124, 125, 126, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 119, 120, 121, 122, 123, 124, 125, 126, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 119, 120, 121, 122, 123, 124, 125, 126, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 119, 120, 121, 122, 123, 124, 125, 126, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 119, 120, 121, 122, 123, 124, 125, 126, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 151, 152, 153, 154, 155, 156, 157, 158, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 151, 152, 153, 154, 155, 156, 157, 158, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 151, 152, 153, 154, 155, 156, 157, 158, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 151, 152, 153, 154, 155, 156, 157, 158, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 151, 152, 153, 154, 155, 156, 157, 158, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 151, 152, 153, 154, 155, 156, 157, 158, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 151, 152, 153, 154, 155, 156, 157, 158, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 151, 152, 153, 154, 155, 156, 157, 158, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 159, 160, 161, 162, 163, 164, 165, 166, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 159, 160, 161, 162, 163, 164, 165, 166, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 159, 160, 161, 162, 163, 164, 165, 166, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 159, 160, 161, 162, 163, 164, 165, 166, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 159, 160, 161, 162, 163, 164, 165, 166, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 159, 160, 161, 162, 163, 164, 165, 166, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 159, 160, 161, 162, 163, 164, 165, 166, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 159, 160, 161, 162, 163, 164, 165, 166, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 623, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 624, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 625, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 626, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 627, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 628, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 629, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 630, 47, 48, 49, 50, 51, 52, 53, 54, 111, 112, 113, 114, 115, 116, 117, 118, 463, 47, 48, 49, 50, 51, 52, 53, 54, 111, 112, 113, 114, 115, 116, 117, 118, 464, 47, 48, 49, 50, 51, 52, 53, 54, 111, 112, 113, 114, 115, 116, 117, 118, 465, 47, 48, 49, 50, 51, 52, 53, 54, 111, 112, 113, 114, 115, 116, 117, 118, 466, 47, 48, 49, 50, 51, 52, 53, 54, 111, 112, 113, 114, 115, 116, 117, 118, 467, 47, 48, 49, 50, 51, 52, 53, 54, 111, 112, 113, 114, 115, 116, 117, 118, 468, 47, 48, 49, 50, 51, 52, 53, 54, 111, 112, 113, 114, 115, 116, 117, 118, 469, 47, 48, 49, 50, 51, 52, 53, 54, 111, 112, 113, 114, 115, 116, 117, 118, 470, 55, 56, 57, 58, 59, 60, 61, 62, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 55, 56, 57, 58, 59, 60, 61, 62, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 55, 56, 57, 58, 59, 60, 61, 62, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 55, 56, 57, 58, 59, 60, 61, 62, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 55, 56, 57, 58, 59, 60, 61, 62, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 55, 56, 57, 58, 59, 60, 61, 62, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 55, 56, 57, 58, 59, 60, 61, 62, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 55, 56, 57, 58, 59, 60, 61, 62, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 634, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 635, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 636, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 637, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 638, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 639, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 640, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 641, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 471, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 472, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 473, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 474, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 475, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 476, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 477, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 478, 71, 72, 73, 74, 75, 76, 77, 78, 151, 152, 153, 154, 155, 156, 157, 158, 479, 71, 72, 73, 74, 75, 76, 77, 78, 151, 152, 153, 154, 155, 156, 157, 158, 480, 71, 72, 73, 74, 75, 76, 77, 78, 151, 152, 153, 154, 155, 156, 157, 158, 481, 71, 72, 73, 74, 75, 76, 77, 78, 151, 152, 153, 154, 155, 156, 157, 158, 482, 71, 72, 73, 74, 75, 76, 77, 78, 151, 152, 153, 154, 155, 156, 157, 158, 483, 71, 72, 73, 74, 75, 76, 77, 78, 151, 152, 153, 154, 155, 156, 157, 158, 484, 71, 72, 73, 74, 75, 76, 77, 78, 151, 152, 153, 154, 155, 156, 157, 158, 485, 71, 72, 73, 74, 75, 76, 77, 78, 151, 152, 153, 154, 155, 156, 157, 158, 486, 95, 96, 97, 98, 99, 100, 101, 102, 159, 160, 161, 162, 163, 164, 165, 166, 487, 95, 96, 97, 98, 99, 100, 101, 102, 159, 160, 161, 162, 163, 164, 165, 166, 488, 95, 96, 97, 98, 99, 100, 101, 102, 159, 160, 161, 162, 163, 164, 165, 166, 489, 95, 96, 97, 98, 99, 100, 101, 102, 159, 160, 161, 162, 163, 164, 165, 166, 490, 95, 96, 97, 98, 99, 100, 101, 102, 159, 160, 161, 162, 163, 164, 165, 166, 491, 95, 96, 97, 98, 99, 100, 101, 102, 159, 160, 161, 162, 163, 164, 165, 166, 492, 95, 96, 97, 98, 99, 100, 101, 102, 159, 160, 161, 162, 163, 164, 165, 166, 493, 95, 96, 97, 98, 99, 100, 101, 102, 159, 160, 161, 162, 163, 164, 165, 166, 494, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 495, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 496, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 497, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 498, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 499, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 500, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 501, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 502, 25, 26, 29, 30, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 589, 25, 26, 29, 30, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 590, 25, 26, 27, 28, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 591, 25, 26, 27, 28, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 592, 27, 28, 29, 30, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 593, 27, 28, 29, 30, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 594, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 595, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 596, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 503, 645, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 504, 646, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 505, 647, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 506, 648, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 507, 649, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 508, 650, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 509, 651, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 510, 652, 25, 26, 29, 30, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 511, 612, 25, 26, 29, 30, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 512, 613, 25, 26, 27, 28, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 513, 614, 25, 26, 27, 28, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 514, 615, 27, 28, 29, 30, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 515, 616, 27, 28, 29, 30, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 516, 617, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 517, 618, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 518, 619, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 263, 264, 265, 266, 267, 268, 269, 270, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 263, 264, 265, 266, 267, 268, 269, 270, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 263, 264, 265, 266, 267, 268, 269, 270, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 263, 264, 265, 266, 267, 268, 269, 270, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 263, 264, 265, 266, 267, 268, 269, 270, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 263, 264, 265, 266, 267, 268, 269, 270, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 263, 264, 265, 266, 267, 268, 269, 270, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 263, 264, 265, 266, 267, 268, 269, 270, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 303, 304, 305, 306, 307, 308, 309, 310, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 303, 304, 305, 306, 307, 308, 309, 310, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 303, 304, 305, 306, 307, 308, 309, 310, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 303, 304, 305, 306, 307, 308, 309, 310, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 303, 304, 305, 306, 307, 308, 309, 310, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 303, 304, 305, 306, 307, 308, 309, 310, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 303, 304, 305, 306, 307, 308, 309, 310, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 303, 304, 305, 306, 307, 308, 309, 310, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 335, 336, 337, 338, 339, 340, 341, 342, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 335, 336, 337, 338, 339, 340, 341, 342, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 335, 336, 337, 338, 339, 340, 341, 342, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 335, 336, 337, 338, 339, 340, 341, 342, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 335, 336, 337, 338, 339, 340, 341, 342, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 335, 336, 337, 338, 339, 340, 341, 342, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 335, 336, 337, 338, 339, 340, 341, 342, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 335, 336, 337, 338, 339, 340, 341, 342, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 656, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 657, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 658, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 659, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 660, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 661, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 662, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 663, 207, 208, 209, 210, 211, 212, 213, 214, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 295, 296, 297, 298, 299, 300, 301, 302, 207, 208, 209, 210, 211, 212, 213, 214, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 295, 296, 297, 298, 299, 300, 301, 302, 207, 208, 209, 210, 211, 212, 213, 214, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 295, 296, 297, 298, 299, 300, 301, 302, 207, 208, 209, 210, 211, 212, 213, 214, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 295, 296, 297, 298, 299, 300, 301, 302, 207, 208, 209, 210, 211, 212, 213, 214, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 295, 296, 297, 298, 299, 300, 301, 302, 207, 208, 209, 210, 211, 212, 213, 214, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 295, 296, 297, 298, 299, 300, 301, 302, 207, 208, 209, 210, 211, 212, 213, 214, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 295, 296, 297, 298, 299, 300, 301, 302, 207, 208, 209, 210, 211, 212, 213, 214, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 295, 296, 297, 298, 299, 300, 301, 302, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 667, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 668, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 669, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 670, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 671, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 672, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 673, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 674, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 519, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 520, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 521, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 522, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 523, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 524, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 525, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 526, 271, 272, 273, 274, 275, 276, 277, 278, 287, 288, 289, 290, 291, 292, 293, 294, 527, 271, 272, 273, 274, 275, 276, 277, 278, 287, 288, 289, 290, 291, 292, 293, 294, 528, 271, 272, 273, 274, 275, 276, 277, 278, 287, 288, 289, 290, 291, 292, 293, 294, 529, 271, 272, 273, 274, 275, 276, 277, 278, 287, 288, 289, 290, 291, 292, 293, 294, 530, 271, 272, 273, 274, 275, 276, 277, 278, 287, 288, 289, 290, 291, 292, 293, 294, 531, 271, 272, 273, 274, 275, 276, 277, 278, 287, 288, 289, 290, 291, 292, 293, 294, 532, 271, 272, 273, 274, 275, 276, 277, 278, 287, 288, 289, 290, 291, 292, 293, 294, 533, 271, 272, 273, 274, 275, 276, 277, 278, 287, 288, 289, 290, 291, 292, 293, 294, 534, 263, 264, 265, 266, 267, 268, 269, 270, 295, 296, 297, 298, 299, 300, 301, 302, 535, 263, 264, 265, 266, 267, 268, 269, 270, 295, 296, 297, 298, 299, 300, 301, 302, 536, 263, 264, 265, 266, 267, 268, 269, 270, 295, 296, 297, 298, 299, 300, 301, 302, 537, 263, 264, 265, 266, 267, 268, 269, 270, 295, 296, 297, 298, 299, 300, 301, 302, 538, 263, 264, 265, 266, 267, 268, 269, 270, 295, 296, 297, 298, 299, 300, 301, 302, 539, 263, 264, 265, 266, 267, 268, 269, 270, 295, 296, 297, 298, 299, 300, 301, 302, 540, 263, 264, 265, 266, 267, 268, 269, 270, 295, 296, 297, 298, 299, 300, 301, 302, 541, 263, 264, 265, 266, 267, 268, 269, 270, 295, 296, 297, 298, 299, 300, 301, 302, 542, 223, 224, 225, 226, 227, 228, 229, 230, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 327, 328, 329, 330, 331, 332, 333, 334, 223, 224, 225, 226, 227, 228, 229, 230, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 327, 328, 329, 330, 331, 332, 333, 334, 223, 224, 225, 226, 227, 228, 229, 230, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 327, 328, 329, 330, 331, 332, 333, 334, 223, 224, 225, 226, 227, 228, 229, 230, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 327, 328, 329, 330, 331, 332, 333, 334, 223, 224, 225, 226, 227, 228, 229, 230, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 327, 328, 329, 330, 331, 332, 333, 334, 223, 224, 225, 226, 227, 228, 229, 230, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 327, 328, 329, 330, 331, 332, 333, 334, 223, 224, 225, 226, 227, 228, 229, 230, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 327, 328, 329, 330, 331, 332, 333, 334, 223, 224, 225, 226, 227, 228, 229, 230, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 327, 328, 329, 330, 331, 332, 333, 334, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 678, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 679, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 680, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 681, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 682, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 683, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 684, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 685, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 543, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 544, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 545, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 546, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 547, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 548, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 549, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 550, 303, 304, 305, 306, 307, 308, 309, 310, 327, 328, 329, 330, 331, 332, 333, 334, 551, 303, 304, 305, 306, 307, 308, 309, 310, 327, 328, 329, 330, 331, 332, 333, 334, 552, 303, 304, 305, 306, 307, 308, 309, 310, 327, 328, 329, 330, 331, 332, 333, 334, 553, 303, 304, 305, 306, 307, 308, 309, 310, 327, 328, 329, 330, 331, 332, 333, 334, 554, 303, 304, 305, 306, 307, 308, 309, 310, 327, 328, 329, 330, 331, 332, 333, 334, 555, 303, 304, 305, 306, 307, 308, 309, 310, 327, 328, 329, 330, 331, 332, 333, 334, 556, 303, 304, 305, 306, 307, 308, 309, 310, 327, 328, 329, 330, 331, 332, 333, 334, 557, 303, 304, 305, 306, 307, 308, 309, 310, 327, 328, 329, 330, 331, 332, 333, 334, 558, 247, 248, 249, 250, 251, 252, 253, 254, 335, 336, 337, 338, 339, 340, 341, 342, 559, 247, 248, 249, 250, 251, 252, 253, 254, 335, 336, 337, 338, 339, 340, 341, 342, 560, 247, 248, 249, 250, 251, 252, 253, 254, 335, 336, 337, 338, 339, 340, 341, 342, 561, 247, 248, 249, 250, 251, 252, 253, 254, 335, 336, 337, 338, 339, 340, 341, 342, 562, 247, 248, 249, 250, 251, 252, 253, 254, 335, 336, 337, 338, 339, 340, 341, 342, 563, 247, 248, 249, 250, 251, 252, 253, 254, 335, 336, 337, 338, 339, 340, 341, 342, 564, 247, 248, 249, 250, 251, 252, 253, 254, 335, 336, 337, 338, 339, 340, 341, 342, 565, 247, 248, 249, 250, 251, 252, 253, 254, 335, 336, 337, 338, 339, 340, 341, 342, 566, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 567, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 568, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 569, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 570, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 571, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 572, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 573, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 574, 351, 355, 609, 352, 356, 353, 357, 610, 354, 358, 351, 355, 359, 379, 352, 356, 360, 380, 353, 357, 361, 381, 354, 358, 362, 382, 355, 359, 363, 395, 356, 360, 364, 396, 357, 361, 365, 397, 358, 362, 366, 398, 359, 363, 367, 383, 360, 364, 368, 384, 361, 365, 369, 385, 362, 366, 370, 386, 363, 367, 371, 407, 364, 368, 372, 408, 365, 369, 373, 409, 366, 370, 374, 410, 367, 371, 375, 387, 368, 372, 376, 388, 369, 373, 377, 389, 370, 374, 378, 390, 371, 375, 391, 631, 372, 376, 392, 373, 377, 393, 632, 374, 378, 394, 355, 379, 576, 356, 380, 357, 381, 576, 358, 382, 363, 383, 577, 364, 384, 365, 385, 577, 366, 386, 371, 387, 578, 372, 388, 373, 389, 578, 374, 390, 375, 391, 579, 376, 392, 377, 393, 579, 378, 394, 359, 395, 399, 642, 360, 396, 400, 361, 397, 401, 643, 362, 398, 402, 395, 399, 575, 396, 400, 397, 401, 575, 398, 402, 403, 407, 597, 404, 408, 405, 409, 598, 406, 410, 367, 403, 407, 447, 580, 653, 368, 404, 408, 448, 369, 405, 409, 449, 580, 654, 370, 406, 410, 450, 411, 415, 620, 412, 416, 413, 417, 621, 414, 418, 411, 415, 419, 431, 412, 416, 420, 432, 413, 417, 421, 433, 414, 418, 422, 434, 415, 419, 423, 439, 416, 420, 424, 440, 417, 421, 425, 441, 418, 422, 426, 442, 419, 423, 427, 447, 420, 424, 428, 448, 421, 425, 429, 449, 422, 426, 430, 450, 423, 427, 451, 664, 424, 428, 452, 425, 429, 453, 665, 426, 430, 454, 415, 431, 435, 675, 416, 432, 436, 417, 433, 437, 676, 418, 434, 438, 431, 435, 581, 432, 436, 433, 437, 581, 434, 438, 419, 439, 443, 686, 420, 440, 444, 421, 441, 445, 687, 422, 442, 446, 439, 443, 582, 440, 444, 441, 445, 582, 442, 446, 407, 423, 447, 583, 408, 424, 448, 409, 425, 449, 583, 410, 426, 450, 427, 451, 584, 428, 452, 429, 453, 584, 430, 454, 31, 32, 37, 38, 455, 456, 33, 34, 37, 38, 457, 458, 35, 36, 37, 38, 459, 460, 31, 32, 37, 38, 455, 456, 33, 34, 37, 38, 457, 458, 35, 36, 37, 38, 459, 460, 455, 457, 459, 461, 456, 458, 460, 462, 111, 112, 117, 118, 463, 464, 113, 114, 117, 118, 465, 466, 115, 116, 117, 118, 467, 468, 111, 112, 117, 118, 463, 464, 113, 114, 117, 118, 465, 466, 115, 116, 117, 118, 467, 468, 463, 465, 467, 469, 464, 466, 468, 470, 143, 144, 149, 150, 471, 472, 145, 146, 149, 150, 473, 474, 147, 148, 149, 150, 475, 476, 143, 144, 149, 150, 471, 472, 145, 146, 149, 150, 473, 474, 147, 148, 149, 150, 475, 476, 471, 473, 475, 477, 472, 474, 476, 478, 151, 152, 157, 158, 479, 480, 153, 154, 157, 158, 481, 482, 155, 156, 157, 158, 483, 484, 151, 152, 157, 158, 479, 480, 153, 154, 157, 158, 481, 482, 155, 156, 157, 158, 483, 484, 479, 481, 483, 485, 480, 482, 484, 486, 159, 160, 165, 166, 487, 488, 161, 162, 165, 166, 489, 490, 163, 164, 165, 166, 491, 492, 159, 160, 165, 166, 487, 488, 161, 162, 165, 166, 489, 490, 163, 164, 165, 166, 491, 492, 487, 489, 491, 493, 488, 490, 492, 494, 167, 168, 173, 174, 495, 496, 169, 170, 173, 174, 497, 498, 171, 172, 173, 174, 499, 500, 167, 168, 173, 174, 495, 496, 169, 170, 173, 174, 497, 498, 171, 172, 173, 174, 499, 500, 495, 497, 499, 501, 496, 498, 500, 502, 183, 184, 189, 190, 503, 504, 185, 186, 189, 190, 505, 506, 187, 188, 189, 190, 507, 508, 183, 184, 189, 190, 503, 504, 185, 186, 189, 190, 505, 506, 187, 188, 189, 190, 507, 508, 503, 505, 507, 509, 504, 506, 508, 510, 191, 192, 197, 198, 511, 512, 193, 194, 197, 198, 513, 514, 195, 196, 197, 198, 515, 516, 191, 192, 197, 198, 511, 512, 193, 194, 197, 198, 513, 514, 195, 196, 197, 198, 515, 516, 511, 513, 515, 517, 512, 514, 516, 518, 279, 280, 285, 286, 519, 520, 281, 282, 285, 286, 521, 522, 283, 284, 285, 286, 523, 524, 279, 280, 285, 286, 519, 520, 281, 282, 285, 286, 521, 522, 283, 284, 285, 286, 523, 524, 519, 521, 523, 525, 520, 522, 524, 526, 287, 288, 293, 294, 527, 528, 289, 290, 293, 294, 529, 530, 291, 292, 293, 294, 531, 532, 287, 288, 293, 294, 527, 528, 289, 290, 293, 294, 529, 530, 291, 292, 293, 294, 531, 532, 527, 529, 531, 533, 528, 530, 532, 534, 295, 296, 301, 302, 535, 536, 297, 298, 301, 302, 537, 538, 299, 300, 301, 302, 539, 540, 295, 296, 301, 302, 535, 536, 297, 298, 301, 302, 537, 538, 299, 300, 301, 302, 539, 540, 535, 537, 539, 541, 536, 538, 540, 542, 319, 320, 325, 326, 543, 544, 321, 322, 325, 326, 545, 546, 323, 324, 325, 326, 547, 548, 319, 320, 325, 326, 543, 544, 321, 322, 325, 326, 545, 546, 323, 324, 325, 326, 547, 548, 543, 545, 547, 549, 544, 546, 548, 550, 327, 328, 333, 334, 551, 552, 329, 330, 333, 334, 553, 554, 331, 332, 333, 334, 555, 556, 327, 328, 333, 334, 551, 552, 329, 330, 333, 334, 553, 554, 331, 332, 333, 334, 555, 556, 551, 553, 555, 557, 552, 554, 556, 558, 335, 336, 341, 342, 559, 560, 337, 338, 341, 342, 561, 562, 339, 340, 341, 342, 563, 564, 335, 336, 341, 342, 559, 560, 337, 338, 341, 342, 561, 562, 339, 340, 341, 342, 563, 564, 559, 561, 563, 565, 560, 562, 564, 566, 343, 344, 349, 350, 567, 568, 345, 346, 349, 350, 569, 570, 347, 348, 349, 350, 571, 572, 343, 344, 349, 350, 567, 568, 345, 346, 349, 350, 569, 570, 347, 348, 349, 350, 571, 572, 567, 569, 571, 573, 568, 570, 572, 574, 399, 401, 575, 379, 381, 576, 383, 385, 577, 387, 389, 578, 391, 393, 579, 407, 409, 580, 435, 437, 581, 443, 445, 582, 447, 449, 583, 451, 453, 584, 585, 600, 586, 600, 587, 600, 181, 182, 588, 595, 596, 175, 176, 181, 182, 585, 589, 590, 595, 596, 175, 176, 181, 182, 589, 590, 177, 178, 181, 182, 586, 591, 592, 595, 596, 177, 178, 181, 182, 591, 592, 179, 180, 181, 182, 587, 593, 594, 595, 596, 179, 180, 181, 182, 593, 594, 589, 591, 593, 595, 590, 592, 594, 596, 403, 597, 599, 405, 598, 599, 597, 598, 599, 597, 598, 600, 1, 31, 32, 37, 38, 601, 602, 31, 32, 37, 38, 601, 602, 2, 33, 34, 37, 38, 603, 604, 33, 34, 37, 38, 603, 604, 3, 35, 36, 37, 38, 605, 606, 35, 36, 37, 38, 605, 606, 601, 603, 605, 607, 602, 604, 606, 608, 351, 353, 609, 611, 351, 353, 610, 611, 1, 2, 3, 601, 602, 603, 604, 605, 606, 607, 608, 611, 13, 191, 192, 197, 198, 612, 613, 191, 192, 197, 198, 612, 613, 14, 193, 194, 197, 198, 614, 615, 193, 194, 197, 198, 614, 615, 15, 195, 196, 197, 198, 616, 617, 195, 196, 197, 198, 616, 617, 612, 614, 616, 618, 613, 615, 617, 619, 411, 413, 620, 622, 411, 413, 621, 622, 13, 14, 15, 612, 613, 614, 615, 616, 617, 618, 619, 622, 4, 103, 104, 109, 110, 623, 624, 103, 104, 109, 110, 623, 624, 5, 105, 106, 109, 110, 625, 626, 105, 106, 109, 110, 625, 626, 6, 107, 108, 109, 110, 627, 628, 107, 108, 109, 110, 627, 628, 623, 625, 627, 629, 624, 626, 628, 630, 375, 377, 631, 633, 375, 377, 632, 633, 4, 5, 6, 623, 624, 625, 626, 627, 628, 629, 630, 633, 7, 135, 136, 141, 142, 634, 635, 135, 136, 141, 142, 634, 635, 8, 137, 138, 141, 142, 636, 637, 137, 138, 141, 142, 636, 637, 9, 139, 140, 141, 142, 638, 639, 139, 140, 141, 142, 638, 639, 634, 636, 638, 640, 635, 637, 639, 641, 395, 397, 642, 644, 395, 397, 643, 644, 7, 8, 9, 634, 635, 636, 637, 638, 639, 640, 641, 644, 10, 183, 184, 189, 190, 645, 646, 183, 184, 189, 190, 645, 646, 11, 185, 186, 189, 190, 647, 648, 185, 186, 189, 190, 647, 648, 12, 187, 188, 189, 190, 649, 650, 187, 188, 189, 190, 649, 650, 645, 647, 649, 651, 646, 648, 650, 652, 407, 409, 653, 655, 407, 409, 654, 655, 10, 11, 12, 645, 646, 647, 648, 649, 650, 651, 652, 655, 16, 255, 256, 261, 262, 656, 657, 255, 256, 261, 262, 656, 657, 17, 257, 258, 261, 262, 658, 659, 257, 258, 261, 262, 658, 659, 18, 259, 260, 261, 262, 660, 661, 259, 260, 261, 262, 660, 661, 656, 658, 660, 662, 657, 659, 661, 663, 427, 429, 664, 666, 427, 429, 665, 666, 16, 17, 18, 656, 657, 658, 659, 660, 661, 662, 663, 666, 19, 271, 272, 277, 278, 667, 668, 271, 272, 277, 278, 667, 668, 20, 273, 274, 277, 278, 669, 670, 273, 274, 277, 278, 669, 670, 21, 275, 276, 277, 278, 671, 672, 275, 276, 277, 278, 671, 672, 667, 669, 671, 673, 668, 670, 672, 674, 431, 433, 675, 677, 431, 433, 676, 677, 19, 20, 21, 667, 668, 669, 670, 671, 672, 673, 674, 677, 22, 311, 312, 317, 318, 678, 679, 311, 312, 317, 318, 678, 679, 23, 313, 314, 317, 318, 680, 681, 313, 314, 317, 318, 680, 681, 24, 315, 316, 317, 318, 682, 683, 315, 316, 317, 318, 682, 683, 678, 680, 682, 684, 679, 681, 683, 685, 439, 441, 686, 688, 439, 441, 687, 688, 22, 23, 24, 678, 679, 680, 681, 682, 683, 684, 685, 688, 25, 689, 692, 27, 690, 693, 29, 691, 694, 26, 689, 692, 28, 690, 693, 30, 691, 694, 695, 0, 695, 696]
    sp_jac_ini_ja = [0, 2, 11, 20, 29, 38, 47, 56, 65, 74, 83, 92, 101, 110, 119, 128, 137, 146, 155, 164, 173, 182, 191, 200, 209, 218, 237, 256, 275, 294, 313, 332, 354, 376, 398, 420, 442, 464, 482, 500, 524, 548, 572, 596, 620, 644, 668, 692, 724, 756, 788, 820, 852, 884, 916, 948, 980, 1012, 1044, 1076, 1108, 1140, 1172, 1204, 1228, 1252, 1276, 1300, 1324, 1348, 1372, 1396, 1428, 1460, 1492, 1524, 1556, 1588, 1620, 1652, 1676, 1700, 1724, 1748, 1772, 1796, 1820, 1844, 1868, 1892, 1916, 1940, 1964, 1988, 2012, 2036, 2068, 2100, 2132, 2164, 2196, 2228, 2260, 2292, 2317, 2342, 2367, 2392, 2417, 2442, 2467, 2492, 2509, 2526, 2543, 2560, 2577, 2594, 2611, 2628, 2652, 2676, 2700, 2724, 2748, 2772, 2796, 2820, 2844, 2868, 2892, 2916, 2940, 2964, 2988, 3012, 3037, 3062, 3087, 3112, 3137, 3162, 3187, 3212, 3229, 3246, 3263, 3280, 3297, 3314, 3331, 3348, 3365, 3382, 3399, 3416, 3433, 3450, 3467, 3484, 3501, 3518, 3535, 3552, 3569, 3586, 3603, 3620, 3637, 3654, 3671, 3688, 3705, 3722, 3739, 3756, 3777, 3798, 3819, 3840, 3861, 3882, 3899, 3916, 3934, 3952, 3970, 3988, 4006, 4024, 4042, 4060, 4082, 4104, 4126, 4148, 4170, 4192, 4210, 4228, 4252, 4276, 4300, 4324, 4348, 4372, 4396, 4420, 4452, 4484, 4516, 4548, 4580, 4612, 4644, 4676, 4700, 4724, 4748, 4772, 4796, 4820, 4844, 4868, 4900, 4932, 4964, 4996, 5028, 5060, 5092, 5124, 5148, 5172, 5196, 5220, 5244, 5268, 5292, 5316, 5340, 5364, 5388, 5412, 5436, 5460, 5484, 5508, 5540, 5572, 5604, 5636, 5668, 5700, 5732, 5764, 5789, 5814, 5839, 5864, 5889, 5914, 5939, 5964, 5996, 6028, 6060, 6092, 6124, 6156, 6188, 6220, 6253, 6286, 6319, 6352, 6385, 6418, 6451, 6484, 6501, 6518, 6535, 6552, 6569, 6586, 6603, 6620, 6637, 6654, 6671, 6688, 6705, 6722, 6739, 6756, 6773, 6790, 6807, 6824, 6841, 6858, 6875, 6892, 6924, 6956, 6988, 7020, 7052, 7084, 7116, 7148, 7173, 7198, 7223, 7248, 7273, 7298, 7323, 7348, 7365, 7382, 7399, 7416, 7433, 7450, 7467, 7484, 7501, 7518, 7535, 7552, 7569, 7586, 7603, 7620, 7637, 7654, 7671, 7688, 7705, 7722, 7739, 7756, 7773, 7790, 7807, 7824, 7841, 7858, 7875, 7892, 7895, 7897, 7900, 7902, 7906, 7910, 7914, 7918, 7922, 7926, 7930, 7934, 7938, 7942, 7946, 7950, 7954, 7958, 7962, 7966, 7970, 7974, 7978, 7982, 7986, 7989, 7993, 7996, 7999, 8001, 8004, 8006, 8009, 8011, 8014, 8016, 8019, 8021, 8024, 8026, 8029, 8031, 8034, 8036, 8040, 8043, 8047, 8050, 8053, 8055, 8058, 8060, 8063, 8065, 8068, 8070, 8076, 8080, 8086, 8090, 8093, 8095, 8098, 8100, 8104, 8108, 8112, 8116, 8120, 8124, 8128, 8132, 8136, 8140, 8144, 8148, 8152, 8155, 8159, 8162, 8166, 8169, 8173, 8176, 8179, 8181, 8184, 8186, 8190, 8193, 8197, 8200, 8203, 8205, 8208, 8210, 8214, 8217, 8221, 8224, 8227, 8229, 8232, 8234, 8240, 8246, 8252, 8258, 8264, 8270, 8274, 8278, 8284, 8290, 8296, 8302, 8308, 8314, 8318, 8322, 8328, 8334, 8340, 8346, 8352, 8358, 8362, 8366, 8372, 8378, 8384, 8390, 8396, 8402, 8406, 8410, 8416, 8422, 8428, 8434, 8440, 8446, 8450, 8454, 8460, 8466, 8472, 8478, 8484, 8490, 8494, 8498, 8504, 8510, 8516, 8522, 8528, 8534, 8538, 8542, 8548, 8554, 8560, 8566, 8572, 8578, 8582, 8586, 8592, 8598, 8604, 8610, 8616, 8622, 8626, 8630, 8636, 8642, 8648, 8654, 8660, 8666, 8670, 8674, 8680, 8686, 8692, 8698, 8704, 8710, 8714, 8718, 8724, 8730, 8736, 8742, 8748, 8754, 8758, 8762, 8768, 8774, 8780, 8786, 8792, 8798, 8802, 8806, 8812, 8818, 8824, 8830, 8836, 8842, 8846, 8850, 8856, 8862, 8868, 8874, 8880, 8886, 8890, 8894, 8897, 8900, 8903, 8906, 8909, 8912, 8915, 8918, 8921, 8924, 8926, 8928, 8930, 8935, 8944, 8950, 8959, 8965, 8974, 8980, 8984, 8988, 8991, 8994, 8997, 9000, 9007, 9013, 9020, 9026, 9033, 9039, 9043, 9047, 9051, 9055, 9067, 9074, 9080, 9087, 9093, 9100, 9106, 9110, 9114, 9118, 9122, 9134, 9141, 9147, 9154, 9160, 9167, 9173, 9177, 9181, 9185, 9189, 9201, 9208, 9214, 9221, 9227, 9234, 9240, 9244, 9248, 9252, 9256, 9268, 9275, 9281, 9288, 9294, 9301, 9307, 9311, 9315, 9319, 9323, 9335, 9342, 9348, 9355, 9361, 9368, 9374, 9378, 9382, 9386, 9390, 9402, 9409, 9415, 9422, 9428, 9435, 9441, 9445, 9449, 9453, 9457, 9469, 9476, 9482, 9489, 9495, 9502, 9508, 9512, 9516, 9520, 9524, 9536, 9539, 9542, 9545, 9548, 9551, 9554, 9555, 9558]
    sp_jac_ini_nia = 697
    sp_jac_ini_nja = 697
    return sp_jac_ini_ia, sp_jac_ini_ja, sp_jac_ini_nia, sp_jac_ini_nja 

def sp_jac_run_vectors():

    sp_jac_run_ia = [0, 695, 1, 31, 32, 37, 38, 351, 352, 353, 354, 2, 33, 34, 37, 38, 351, 352, 353, 354, 3, 35, 36, 37, 38, 351, 352, 353, 354, 4, 103, 104, 109, 110, 375, 376, 377, 378, 5, 105, 106, 109, 110, 375, 376, 377, 378, 6, 107, 108, 109, 110, 375, 376, 377, 378, 7, 135, 136, 141, 142, 395, 396, 397, 398, 8, 137, 138, 141, 142, 395, 396, 397, 398, 9, 139, 140, 141, 142, 395, 396, 397, 398, 10, 183, 184, 189, 190, 407, 408, 409, 410, 11, 185, 186, 189, 190, 407, 408, 409, 410, 12, 187, 188, 189, 190, 407, 408, 409, 410, 13, 191, 192, 197, 198, 411, 412, 413, 414, 14, 193, 194, 197, 198, 411, 412, 413, 414, 15, 195, 196, 197, 198, 411, 412, 413, 414, 16, 255, 256, 261, 262, 427, 428, 429, 430, 17, 257, 258, 261, 262, 427, 428, 429, 430, 18, 259, 260, 261, 262, 427, 428, 429, 430, 19, 271, 272, 277, 278, 431, 432, 433, 434, 20, 273, 274, 277, 278, 431, 432, 433, 434, 21, 275, 276, 277, 278, 431, 432, 433, 434, 22, 311, 312, 317, 318, 439, 440, 441, 442, 23, 313, 314, 317, 318, 439, 440, 441, 442, 24, 315, 316, 317, 318, 439, 440, 441, 442, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 175, 176, 177, 178, 191, 192, 193, 194, 689, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 175, 176, 177, 178, 191, 192, 193, 194, 692, 25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 177, 178, 179, 180, 193, 194, 195, 196, 690, 25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 177, 178, 179, 180, 193, 194, 195, 196, 693, 25, 26, 27, 28, 29, 30, 31, 32, 35, 36, 175, 176, 179, 180, 191, 192, 195, 196, 691, 25, 26, 27, 28, 29, 30, 31, 32, 35, 36, 175, 176, 179, 180, 191, 192, 195, 196, 694, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 455, 601, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 456, 602, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 457, 603, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 458, 604, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 459, 605, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 460, 606, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 461, 607, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 462, 608, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117, 118, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117, 118, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117, 118, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117, 118, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117, 118, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117, 118, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117, 118, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117, 118, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 119, 120, 121, 122, 123, 124, 125, 126, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 119, 120, 121, 122, 123, 124, 125, 126, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 119, 120, 121, 122, 123, 124, 125, 126, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 119, 120, 121, 122, 123, 124, 125, 126, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 119, 120, 121, 122, 123, 124, 125, 126, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 119, 120, 121, 122, 123, 124, 125, 126, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 119, 120, 121, 122, 123, 124, 125, 126, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 119, 120, 121, 122, 123, 124, 125, 126, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 151, 152, 153, 154, 155, 156, 157, 158, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 151, 152, 153, 154, 155, 156, 157, 158, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 151, 152, 153, 154, 155, 156, 157, 158, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 151, 152, 153, 154, 155, 156, 157, 158, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 151, 152, 153, 154, 155, 156, 157, 158, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 151, 152, 153, 154, 155, 156, 157, 158, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 151, 152, 153, 154, 155, 156, 157, 158, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 151, 152, 153, 154, 155, 156, 157, 158, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 159, 160, 161, 162, 163, 164, 165, 166, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 159, 160, 161, 162, 163, 164, 165, 166, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 159, 160, 161, 162, 163, 164, 165, 166, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 159, 160, 161, 162, 163, 164, 165, 166, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 159, 160, 161, 162, 163, 164, 165, 166, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 159, 160, 161, 162, 163, 164, 165, 166, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 159, 160, 161, 162, 163, 164, 165, 166, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 159, 160, 161, 162, 163, 164, 165, 166, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 623, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 624, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 625, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 626, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 627, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 628, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 629, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 630, 47, 48, 49, 50, 51, 52, 53, 54, 111, 112, 113, 114, 115, 116, 117, 118, 463, 47, 48, 49, 50, 51, 52, 53, 54, 111, 112, 113, 114, 115, 116, 117, 118, 464, 47, 48, 49, 50, 51, 52, 53, 54, 111, 112, 113, 114, 115, 116, 117, 118, 465, 47, 48, 49, 50, 51, 52, 53, 54, 111, 112, 113, 114, 115, 116, 117, 118, 466, 47, 48, 49, 50, 51, 52, 53, 54, 111, 112, 113, 114, 115, 116, 117, 118, 467, 47, 48, 49, 50, 51, 52, 53, 54, 111, 112, 113, 114, 115, 116, 117, 118, 468, 47, 48, 49, 50, 51, 52, 53, 54, 111, 112, 113, 114, 115, 116, 117, 118, 469, 47, 48, 49, 50, 51, 52, 53, 54, 111, 112, 113, 114, 115, 116, 117, 118, 470, 55, 56, 57, 58, 59, 60, 61, 62, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 55, 56, 57, 58, 59, 60, 61, 62, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 55, 56, 57, 58, 59, 60, 61, 62, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 55, 56, 57, 58, 59, 60, 61, 62, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 55, 56, 57, 58, 59, 60, 61, 62, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 55, 56, 57, 58, 59, 60, 61, 62, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 55, 56, 57, 58, 59, 60, 61, 62, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 55, 56, 57, 58, 59, 60, 61, 62, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 634, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 635, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 636, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 637, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 638, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 639, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 640, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 641, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 471, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 472, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 473, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 474, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 475, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 476, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 477, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 478, 71, 72, 73, 74, 75, 76, 77, 78, 151, 152, 153, 154, 155, 156, 157, 158, 479, 71, 72, 73, 74, 75, 76, 77, 78, 151, 152, 153, 154, 155, 156, 157, 158, 480, 71, 72, 73, 74, 75, 76, 77, 78, 151, 152, 153, 154, 155, 156, 157, 158, 481, 71, 72, 73, 74, 75, 76, 77, 78, 151, 152, 153, 154, 155, 156, 157, 158, 482, 71, 72, 73, 74, 75, 76, 77, 78, 151, 152, 153, 154, 155, 156, 157, 158, 483, 71, 72, 73, 74, 75, 76, 77, 78, 151, 152, 153, 154, 155, 156, 157, 158, 484, 71, 72, 73, 74, 75, 76, 77, 78, 151, 152, 153, 154, 155, 156, 157, 158, 485, 71, 72, 73, 74, 75, 76, 77, 78, 151, 152, 153, 154, 155, 156, 157, 158, 486, 95, 96, 97, 98, 99, 100, 101, 102, 159, 160, 161, 162, 163, 164, 165, 166, 487, 95, 96, 97, 98, 99, 100, 101, 102, 159, 160, 161, 162, 163, 164, 165, 166, 488, 95, 96, 97, 98, 99, 100, 101, 102, 159, 160, 161, 162, 163, 164, 165, 166, 489, 95, 96, 97, 98, 99, 100, 101, 102, 159, 160, 161, 162, 163, 164, 165, 166, 490, 95, 96, 97, 98, 99, 100, 101, 102, 159, 160, 161, 162, 163, 164, 165, 166, 491, 95, 96, 97, 98, 99, 100, 101, 102, 159, 160, 161, 162, 163, 164, 165, 166, 492, 95, 96, 97, 98, 99, 100, 101, 102, 159, 160, 161, 162, 163, 164, 165, 166, 493, 95, 96, 97, 98, 99, 100, 101, 102, 159, 160, 161, 162, 163, 164, 165, 166, 494, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 495, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 496, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 497, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 498, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 499, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 500, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 501, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 502, 25, 26, 29, 30, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 589, 25, 26, 29, 30, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 590, 25, 26, 27, 28, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 591, 25, 26, 27, 28, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 592, 27, 28, 29, 30, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 593, 27, 28, 29, 30, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 594, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 595, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 596, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 503, 645, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 504, 646, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 505, 647, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 506, 648, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 507, 649, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 508, 650, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 509, 651, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 510, 652, 25, 26, 29, 30, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 511, 612, 25, 26, 29, 30, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 512, 613, 25, 26, 27, 28, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 513, 614, 25, 26, 27, 28, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 514, 615, 27, 28, 29, 30, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 515, 616, 27, 28, 29, 30, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 516, 617, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 517, 618, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 518, 619, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 263, 264, 265, 266, 267, 268, 269, 270, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 263, 264, 265, 266, 267, 268, 269, 270, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 263, 264, 265, 266, 267, 268, 269, 270, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 263, 264, 265, 266, 267, 268, 269, 270, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 263, 264, 265, 266, 267, 268, 269, 270, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 263, 264, 265, 266, 267, 268, 269, 270, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 263, 264, 265, 266, 267, 268, 269, 270, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 263, 264, 265, 266, 267, 268, 269, 270, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 303, 304, 305, 306, 307, 308, 309, 310, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 303, 304, 305, 306, 307, 308, 309, 310, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 303, 304, 305, 306, 307, 308, 309, 310, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 303, 304, 305, 306, 307, 308, 309, 310, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 303, 304, 305, 306, 307, 308, 309, 310, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 303, 304, 305, 306, 307, 308, 309, 310, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 303, 304, 305, 306, 307, 308, 309, 310, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 303, 304, 305, 306, 307, 308, 309, 310, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 335, 336, 337, 338, 339, 340, 341, 342, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 335, 336, 337, 338, 339, 340, 341, 342, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 335, 336, 337, 338, 339, 340, 341, 342, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 335, 336, 337, 338, 339, 340, 341, 342, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 335, 336, 337, 338, 339, 340, 341, 342, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 335, 336, 337, 338, 339, 340, 341, 342, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 335, 336, 337, 338, 339, 340, 341, 342, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 335, 336, 337, 338, 339, 340, 341, 342, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 656, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 657, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 658, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 659, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 660, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 661, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 662, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 663, 207, 208, 209, 210, 211, 212, 213, 214, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 295, 296, 297, 298, 299, 300, 301, 302, 207, 208, 209, 210, 211, 212, 213, 214, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 295, 296, 297, 298, 299, 300, 301, 302, 207, 208, 209, 210, 211, 212, 213, 214, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 295, 296, 297, 298, 299, 300, 301, 302, 207, 208, 209, 210, 211, 212, 213, 214, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 295, 296, 297, 298, 299, 300, 301, 302, 207, 208, 209, 210, 211, 212, 213, 214, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 295, 296, 297, 298, 299, 300, 301, 302, 207, 208, 209, 210, 211, 212, 213, 214, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 295, 296, 297, 298, 299, 300, 301, 302, 207, 208, 209, 210, 211, 212, 213, 214, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 295, 296, 297, 298, 299, 300, 301, 302, 207, 208, 209, 210, 211, 212, 213, 214, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 295, 296, 297, 298, 299, 300, 301, 302, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 667, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 668, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 669, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 670, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 671, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 672, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 673, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 674, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 519, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 520, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 521, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 522, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 523, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 524, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 525, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 526, 271, 272, 273, 274, 275, 276, 277, 278, 287, 288, 289, 290, 291, 292, 293, 294, 527, 271, 272, 273, 274, 275, 276, 277, 278, 287, 288, 289, 290, 291, 292, 293, 294, 528, 271, 272, 273, 274, 275, 276, 277, 278, 287, 288, 289, 290, 291, 292, 293, 294, 529, 271, 272, 273, 274, 275, 276, 277, 278, 287, 288, 289, 290, 291, 292, 293, 294, 530, 271, 272, 273, 274, 275, 276, 277, 278, 287, 288, 289, 290, 291, 292, 293, 294, 531, 271, 272, 273, 274, 275, 276, 277, 278, 287, 288, 289, 290, 291, 292, 293, 294, 532, 271, 272, 273, 274, 275, 276, 277, 278, 287, 288, 289, 290, 291, 292, 293, 294, 533, 271, 272, 273, 274, 275, 276, 277, 278, 287, 288, 289, 290, 291, 292, 293, 294, 534, 263, 264, 265, 266, 267, 268, 269, 270, 295, 296, 297, 298, 299, 300, 301, 302, 535, 263, 264, 265, 266, 267, 268, 269, 270, 295, 296, 297, 298, 299, 300, 301, 302, 536, 263, 264, 265, 266, 267, 268, 269, 270, 295, 296, 297, 298, 299, 300, 301, 302, 537, 263, 264, 265, 266, 267, 268, 269, 270, 295, 296, 297, 298, 299, 300, 301, 302, 538, 263, 264, 265, 266, 267, 268, 269, 270, 295, 296, 297, 298, 299, 300, 301, 302, 539, 263, 264, 265, 266, 267, 268, 269, 270, 295, 296, 297, 298, 299, 300, 301, 302, 540, 263, 264, 265, 266, 267, 268, 269, 270, 295, 296, 297, 298, 299, 300, 301, 302, 541, 263, 264, 265, 266, 267, 268, 269, 270, 295, 296, 297, 298, 299, 300, 301, 302, 542, 223, 224, 225, 226, 227, 228, 229, 230, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 327, 328, 329, 330, 331, 332, 333, 334, 223, 224, 225, 226, 227, 228, 229, 230, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 327, 328, 329, 330, 331, 332, 333, 334, 223, 224, 225, 226, 227, 228, 229, 230, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 327, 328, 329, 330, 331, 332, 333, 334, 223, 224, 225, 226, 227, 228, 229, 230, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 327, 328, 329, 330, 331, 332, 333, 334, 223, 224, 225, 226, 227, 228, 229, 230, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 327, 328, 329, 330, 331, 332, 333, 334, 223, 224, 225, 226, 227, 228, 229, 230, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 327, 328, 329, 330, 331, 332, 333, 334, 223, 224, 225, 226, 227, 228, 229, 230, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 327, 328, 329, 330, 331, 332, 333, 334, 223, 224, 225, 226, 227, 228, 229, 230, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 327, 328, 329, 330, 331, 332, 333, 334, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 678, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 679, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 680, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 681, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 682, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 683, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 684, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 685, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 543, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 544, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 545, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 546, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 547, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 548, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 549, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 550, 303, 304, 305, 306, 307, 308, 309, 310, 327, 328, 329, 330, 331, 332, 333, 334, 551, 303, 304, 305, 306, 307, 308, 309, 310, 327, 328, 329, 330, 331, 332, 333, 334, 552, 303, 304, 305, 306, 307, 308, 309, 310, 327, 328, 329, 330, 331, 332, 333, 334, 553, 303, 304, 305, 306, 307, 308, 309, 310, 327, 328, 329, 330, 331, 332, 333, 334, 554, 303, 304, 305, 306, 307, 308, 309, 310, 327, 328, 329, 330, 331, 332, 333, 334, 555, 303, 304, 305, 306, 307, 308, 309, 310, 327, 328, 329, 330, 331, 332, 333, 334, 556, 303, 304, 305, 306, 307, 308, 309, 310, 327, 328, 329, 330, 331, 332, 333, 334, 557, 303, 304, 305, 306, 307, 308, 309, 310, 327, 328, 329, 330, 331, 332, 333, 334, 558, 247, 248, 249, 250, 251, 252, 253, 254, 335, 336, 337, 338, 339, 340, 341, 342, 559, 247, 248, 249, 250, 251, 252, 253, 254, 335, 336, 337, 338, 339, 340, 341, 342, 560, 247, 248, 249, 250, 251, 252, 253, 254, 335, 336, 337, 338, 339, 340, 341, 342, 561, 247, 248, 249, 250, 251, 252, 253, 254, 335, 336, 337, 338, 339, 340, 341, 342, 562, 247, 248, 249, 250, 251, 252, 253, 254, 335, 336, 337, 338, 339, 340, 341, 342, 563, 247, 248, 249, 250, 251, 252, 253, 254, 335, 336, 337, 338, 339, 340, 341, 342, 564, 247, 248, 249, 250, 251, 252, 253, 254, 335, 336, 337, 338, 339, 340, 341, 342, 565, 247, 248, 249, 250, 251, 252, 253, 254, 335, 336, 337, 338, 339, 340, 341, 342, 566, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 567, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 568, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 569, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 570, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 571, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 572, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 573, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 574, 351, 355, 609, 352, 356, 353, 357, 610, 354, 358, 351, 355, 359, 379, 352, 356, 360, 380, 353, 357, 361, 381, 354, 358, 362, 382, 355, 359, 363, 395, 356, 360, 364, 396, 357, 361, 365, 397, 358, 362, 366, 398, 359, 363, 367, 383, 360, 364, 368, 384, 361, 365, 369, 385, 362, 366, 370, 386, 363, 367, 371, 407, 364, 368, 372, 408, 365, 369, 373, 409, 366, 370, 374, 410, 367, 371, 375, 387, 368, 372, 376, 388, 369, 373, 377, 389, 370, 374, 378, 390, 371, 375, 391, 631, 372, 376, 392, 373, 377, 393, 632, 374, 378, 394, 355, 379, 576, 356, 380, 357, 381, 576, 358, 382, 363, 383, 577, 364, 384, 365, 385, 577, 366, 386, 371, 387, 578, 372, 388, 373, 389, 578, 374, 390, 375, 391, 579, 376, 392, 377, 393, 579, 378, 394, 359, 395, 399, 642, 360, 396, 400, 361, 397, 401, 643, 362, 398, 402, 395, 399, 575, 396, 400, 397, 401, 575, 398, 402, 403, 407, 597, 404, 408, 405, 409, 598, 406, 410, 367, 403, 407, 447, 580, 653, 368, 404, 408, 448, 369, 405, 409, 449, 580, 654, 370, 406, 410, 450, 411, 415, 620, 412, 416, 413, 417, 621, 414, 418, 411, 415, 419, 431, 412, 416, 420, 432, 413, 417, 421, 433, 414, 418, 422, 434, 415, 419, 423, 439, 416, 420, 424, 440, 417, 421, 425, 441, 418, 422, 426, 442, 419, 423, 427, 447, 420, 424, 428, 448, 421, 425, 429, 449, 422, 426, 430, 450, 423, 427, 451, 664, 424, 428, 452, 425, 429, 453, 665, 426, 430, 454, 415, 431, 435, 675, 416, 432, 436, 417, 433, 437, 676, 418, 434, 438, 431, 435, 581, 432, 436, 433, 437, 581, 434, 438, 419, 439, 443, 686, 420, 440, 444, 421, 441, 445, 687, 422, 442, 446, 439, 443, 582, 440, 444, 441, 445, 582, 442, 446, 407, 423, 447, 583, 408, 424, 448, 409, 425, 449, 583, 410, 426, 450, 427, 451, 584, 428, 452, 429, 453, 584, 430, 454, 31, 32, 37, 38, 455, 456, 33, 34, 37, 38, 457, 458, 35, 36, 37, 38, 459, 460, 31, 32, 37, 38, 455, 456, 33, 34, 37, 38, 457, 458, 35, 36, 37, 38, 459, 460, 455, 457, 459, 461, 456, 458, 460, 462, 111, 112, 117, 118, 463, 464, 113, 114, 117, 118, 465, 466, 115, 116, 117, 118, 467, 468, 111, 112, 117, 118, 463, 464, 113, 114, 117, 118, 465, 466, 115, 116, 117, 118, 467, 468, 463, 465, 467, 469, 464, 466, 468, 470, 143, 144, 149, 150, 471, 472, 145, 146, 149, 150, 473, 474, 147, 148, 149, 150, 475, 476, 143, 144, 149, 150, 471, 472, 145, 146, 149, 150, 473, 474, 147, 148, 149, 150, 475, 476, 471, 473, 475, 477, 472, 474, 476, 478, 151, 152, 157, 158, 479, 480, 153, 154, 157, 158, 481, 482, 155, 156, 157, 158, 483, 484, 151, 152, 157, 158, 479, 480, 153, 154, 157, 158, 481, 482, 155, 156, 157, 158, 483, 484, 479, 481, 483, 485, 480, 482, 484, 486, 159, 160, 165, 166, 487, 488, 161, 162, 165, 166, 489, 490, 163, 164, 165, 166, 491, 492, 159, 160, 165, 166, 487, 488, 161, 162, 165, 166, 489, 490, 163, 164, 165, 166, 491, 492, 487, 489, 491, 493, 488, 490, 492, 494, 167, 168, 173, 174, 495, 496, 169, 170, 173, 174, 497, 498, 171, 172, 173, 174, 499, 500, 167, 168, 173, 174, 495, 496, 169, 170, 173, 174, 497, 498, 171, 172, 173, 174, 499, 500, 495, 497, 499, 501, 496, 498, 500, 502, 183, 184, 189, 190, 503, 504, 185, 186, 189, 190, 505, 506, 187, 188, 189, 190, 507, 508, 183, 184, 189, 190, 503, 504, 185, 186, 189, 190, 505, 506, 187, 188, 189, 190, 507, 508, 503, 505, 507, 509, 504, 506, 508, 510, 191, 192, 197, 198, 511, 512, 193, 194, 197, 198, 513, 514, 195, 196, 197, 198, 515, 516, 191, 192, 197, 198, 511, 512, 193, 194, 197, 198, 513, 514, 195, 196, 197, 198, 515, 516, 511, 513, 515, 517, 512, 514, 516, 518, 279, 280, 285, 286, 519, 520, 281, 282, 285, 286, 521, 522, 283, 284, 285, 286, 523, 524, 279, 280, 285, 286, 519, 520, 281, 282, 285, 286, 521, 522, 283, 284, 285, 286, 523, 524, 519, 521, 523, 525, 520, 522, 524, 526, 287, 288, 293, 294, 527, 528, 289, 290, 293, 294, 529, 530, 291, 292, 293, 294, 531, 532, 287, 288, 293, 294, 527, 528, 289, 290, 293, 294, 529, 530, 291, 292, 293, 294, 531, 532, 527, 529, 531, 533, 528, 530, 532, 534, 295, 296, 301, 302, 535, 536, 297, 298, 301, 302, 537, 538, 299, 300, 301, 302, 539, 540, 295, 296, 301, 302, 535, 536, 297, 298, 301, 302, 537, 538, 299, 300, 301, 302, 539, 540, 535, 537, 539, 541, 536, 538, 540, 542, 319, 320, 325, 326, 543, 544, 321, 322, 325, 326, 545, 546, 323, 324, 325, 326, 547, 548, 319, 320, 325, 326, 543, 544, 321, 322, 325, 326, 545, 546, 323, 324, 325, 326, 547, 548, 543, 545, 547, 549, 544, 546, 548, 550, 327, 328, 333, 334, 551, 552, 329, 330, 333, 334, 553, 554, 331, 332, 333, 334, 555, 556, 327, 328, 333, 334, 551, 552, 329, 330, 333, 334, 553, 554, 331, 332, 333, 334, 555, 556, 551, 553, 555, 557, 552, 554, 556, 558, 335, 336, 341, 342, 559, 560, 337, 338, 341, 342, 561, 562, 339, 340, 341, 342, 563, 564, 335, 336, 341, 342, 559, 560, 337, 338, 341, 342, 561, 562, 339, 340, 341, 342, 563, 564, 559, 561, 563, 565, 560, 562, 564, 566, 343, 344, 349, 350, 567, 568, 345, 346, 349, 350, 569, 570, 347, 348, 349, 350, 571, 572, 343, 344, 349, 350, 567, 568, 345, 346, 349, 350, 569, 570, 347, 348, 349, 350, 571, 572, 567, 569, 571, 573, 568, 570, 572, 574, 399, 401, 575, 379, 381, 576, 383, 385, 577, 387, 389, 578, 391, 393, 579, 407, 409, 580, 435, 437, 581, 443, 445, 582, 447, 449, 583, 451, 453, 584, 585, 600, 586, 600, 587, 600, 181, 182, 588, 595, 596, 175, 176, 181, 182, 585, 589, 590, 595, 596, 175, 176, 181, 182, 589, 590, 177, 178, 181, 182, 586, 591, 592, 595, 596, 177, 178, 181, 182, 591, 592, 179, 180, 181, 182, 587, 593, 594, 595, 596, 179, 180, 181, 182, 593, 594, 589, 591, 593, 595, 590, 592, 594, 596, 403, 597, 599, 405, 598, 599, 597, 598, 599, 597, 598, 600, 1, 31, 32, 37, 38, 601, 602, 31, 32, 37, 38, 601, 602, 2, 33, 34, 37, 38, 603, 604, 33, 34, 37, 38, 603, 604, 3, 35, 36, 37, 38, 605, 606, 35, 36, 37, 38, 605, 606, 601, 603, 605, 607, 602, 604, 606, 608, 351, 353, 609, 611, 351, 353, 610, 611, 1, 2, 3, 601, 602, 603, 604, 605, 606, 607, 608, 611, 13, 191, 192, 197, 198, 612, 613, 191, 192, 197, 198, 612, 613, 14, 193, 194, 197, 198, 614, 615, 193, 194, 197, 198, 614, 615, 15, 195, 196, 197, 198, 616, 617, 195, 196, 197, 198, 616, 617, 612, 614, 616, 618, 613, 615, 617, 619, 411, 413, 620, 622, 411, 413, 621, 622, 13, 14, 15, 612, 613, 614, 615, 616, 617, 618, 619, 622, 4, 103, 104, 109, 110, 623, 624, 103, 104, 109, 110, 623, 624, 5, 105, 106, 109, 110, 625, 626, 105, 106, 109, 110, 625, 626, 6, 107, 108, 109, 110, 627, 628, 107, 108, 109, 110, 627, 628, 623, 625, 627, 629, 624, 626, 628, 630, 375, 377, 631, 633, 375, 377, 632, 633, 4, 5, 6, 623, 624, 625, 626, 627, 628, 629, 630, 633, 7, 135, 136, 141, 142, 634, 635, 135, 136, 141, 142, 634, 635, 8, 137, 138, 141, 142, 636, 637, 137, 138, 141, 142, 636, 637, 9, 139, 140, 141, 142, 638, 639, 139, 140, 141, 142, 638, 639, 634, 636, 638, 640, 635, 637, 639, 641, 395, 397, 642, 644, 395, 397, 643, 644, 7, 8, 9, 634, 635, 636, 637, 638, 639, 640, 641, 644, 10, 183, 184, 189, 190, 645, 646, 183, 184, 189, 190, 645, 646, 11, 185, 186, 189, 190, 647, 648, 185, 186, 189, 190, 647, 648, 12, 187, 188, 189, 190, 649, 650, 187, 188, 189, 190, 649, 650, 645, 647, 649, 651, 646, 648, 650, 652, 407, 409, 653, 655, 407, 409, 654, 655, 10, 11, 12, 645, 646, 647, 648, 649, 650, 651, 652, 655, 16, 255, 256, 261, 262, 656, 657, 255, 256, 261, 262, 656, 657, 17, 257, 258, 261, 262, 658, 659, 257, 258, 261, 262, 658, 659, 18, 259, 260, 261, 262, 660, 661, 259, 260, 261, 262, 660, 661, 656, 658, 660, 662, 657, 659, 661, 663, 427, 429, 664, 666, 427, 429, 665, 666, 16, 17, 18, 656, 657, 658, 659, 660, 661, 662, 663, 666, 19, 271, 272, 277, 278, 667, 668, 271, 272, 277, 278, 667, 668, 20, 273, 274, 277, 278, 669, 670, 273, 274, 277, 278, 669, 670, 21, 275, 276, 277, 278, 671, 672, 275, 276, 277, 278, 671, 672, 667, 669, 671, 673, 668, 670, 672, 674, 431, 433, 675, 677, 431, 433, 676, 677, 19, 20, 21, 667, 668, 669, 670, 671, 672, 673, 674, 677, 22, 311, 312, 317, 318, 678, 679, 311, 312, 317, 318, 678, 679, 23, 313, 314, 317, 318, 680, 681, 313, 314, 317, 318, 680, 681, 24, 315, 316, 317, 318, 682, 683, 315, 316, 317, 318, 682, 683, 678, 680, 682, 684, 679, 681, 683, 685, 439, 441, 686, 688, 439, 441, 687, 688, 22, 23, 24, 678, 679, 680, 681, 682, 683, 684, 685, 688, 25, 689, 692, 27, 690, 693, 29, 691, 694, 26, 689, 692, 28, 690, 693, 30, 691, 694, 695, 0, 695, 696]
    sp_jac_run_ja = [0, 2, 11, 20, 29, 38, 47, 56, 65, 74, 83, 92, 101, 110, 119, 128, 137, 146, 155, 164, 173, 182, 191, 200, 209, 218, 237, 256, 275, 294, 313, 332, 354, 376, 398, 420, 442, 464, 482, 500, 524, 548, 572, 596, 620, 644, 668, 692, 724, 756, 788, 820, 852, 884, 916, 948, 980, 1012, 1044, 1076, 1108, 1140, 1172, 1204, 1228, 1252, 1276, 1300, 1324, 1348, 1372, 1396, 1428, 1460, 1492, 1524, 1556, 1588, 1620, 1652, 1676, 1700, 1724, 1748, 1772, 1796, 1820, 1844, 1868, 1892, 1916, 1940, 1964, 1988, 2012, 2036, 2068, 2100, 2132, 2164, 2196, 2228, 2260, 2292, 2317, 2342, 2367, 2392, 2417, 2442, 2467, 2492, 2509, 2526, 2543, 2560, 2577, 2594, 2611, 2628, 2652, 2676, 2700, 2724, 2748, 2772, 2796, 2820, 2844, 2868, 2892, 2916, 2940, 2964, 2988, 3012, 3037, 3062, 3087, 3112, 3137, 3162, 3187, 3212, 3229, 3246, 3263, 3280, 3297, 3314, 3331, 3348, 3365, 3382, 3399, 3416, 3433, 3450, 3467, 3484, 3501, 3518, 3535, 3552, 3569, 3586, 3603, 3620, 3637, 3654, 3671, 3688, 3705, 3722, 3739, 3756, 3777, 3798, 3819, 3840, 3861, 3882, 3899, 3916, 3934, 3952, 3970, 3988, 4006, 4024, 4042, 4060, 4082, 4104, 4126, 4148, 4170, 4192, 4210, 4228, 4252, 4276, 4300, 4324, 4348, 4372, 4396, 4420, 4452, 4484, 4516, 4548, 4580, 4612, 4644, 4676, 4700, 4724, 4748, 4772, 4796, 4820, 4844, 4868, 4900, 4932, 4964, 4996, 5028, 5060, 5092, 5124, 5148, 5172, 5196, 5220, 5244, 5268, 5292, 5316, 5340, 5364, 5388, 5412, 5436, 5460, 5484, 5508, 5540, 5572, 5604, 5636, 5668, 5700, 5732, 5764, 5789, 5814, 5839, 5864, 5889, 5914, 5939, 5964, 5996, 6028, 6060, 6092, 6124, 6156, 6188, 6220, 6253, 6286, 6319, 6352, 6385, 6418, 6451, 6484, 6501, 6518, 6535, 6552, 6569, 6586, 6603, 6620, 6637, 6654, 6671, 6688, 6705, 6722, 6739, 6756, 6773, 6790, 6807, 6824, 6841, 6858, 6875, 6892, 6924, 6956, 6988, 7020, 7052, 7084, 7116, 7148, 7173, 7198, 7223, 7248, 7273, 7298, 7323, 7348, 7365, 7382, 7399, 7416, 7433, 7450, 7467, 7484, 7501, 7518, 7535, 7552, 7569, 7586, 7603, 7620, 7637, 7654, 7671, 7688, 7705, 7722, 7739, 7756, 7773, 7790, 7807, 7824, 7841, 7858, 7875, 7892, 7895, 7897, 7900, 7902, 7906, 7910, 7914, 7918, 7922, 7926, 7930, 7934, 7938, 7942, 7946, 7950, 7954, 7958, 7962, 7966, 7970, 7974, 7978, 7982, 7986, 7989, 7993, 7996, 7999, 8001, 8004, 8006, 8009, 8011, 8014, 8016, 8019, 8021, 8024, 8026, 8029, 8031, 8034, 8036, 8040, 8043, 8047, 8050, 8053, 8055, 8058, 8060, 8063, 8065, 8068, 8070, 8076, 8080, 8086, 8090, 8093, 8095, 8098, 8100, 8104, 8108, 8112, 8116, 8120, 8124, 8128, 8132, 8136, 8140, 8144, 8148, 8152, 8155, 8159, 8162, 8166, 8169, 8173, 8176, 8179, 8181, 8184, 8186, 8190, 8193, 8197, 8200, 8203, 8205, 8208, 8210, 8214, 8217, 8221, 8224, 8227, 8229, 8232, 8234, 8240, 8246, 8252, 8258, 8264, 8270, 8274, 8278, 8284, 8290, 8296, 8302, 8308, 8314, 8318, 8322, 8328, 8334, 8340, 8346, 8352, 8358, 8362, 8366, 8372, 8378, 8384, 8390, 8396, 8402, 8406, 8410, 8416, 8422, 8428, 8434, 8440, 8446, 8450, 8454, 8460, 8466, 8472, 8478, 8484, 8490, 8494, 8498, 8504, 8510, 8516, 8522, 8528, 8534, 8538, 8542, 8548, 8554, 8560, 8566, 8572, 8578, 8582, 8586, 8592, 8598, 8604, 8610, 8616, 8622, 8626, 8630, 8636, 8642, 8648, 8654, 8660, 8666, 8670, 8674, 8680, 8686, 8692, 8698, 8704, 8710, 8714, 8718, 8724, 8730, 8736, 8742, 8748, 8754, 8758, 8762, 8768, 8774, 8780, 8786, 8792, 8798, 8802, 8806, 8812, 8818, 8824, 8830, 8836, 8842, 8846, 8850, 8856, 8862, 8868, 8874, 8880, 8886, 8890, 8894, 8897, 8900, 8903, 8906, 8909, 8912, 8915, 8918, 8921, 8924, 8926, 8928, 8930, 8935, 8944, 8950, 8959, 8965, 8974, 8980, 8984, 8988, 8991, 8994, 8997, 9000, 9007, 9013, 9020, 9026, 9033, 9039, 9043, 9047, 9051, 9055, 9067, 9074, 9080, 9087, 9093, 9100, 9106, 9110, 9114, 9118, 9122, 9134, 9141, 9147, 9154, 9160, 9167, 9173, 9177, 9181, 9185, 9189, 9201, 9208, 9214, 9221, 9227, 9234, 9240, 9244, 9248, 9252, 9256, 9268, 9275, 9281, 9288, 9294, 9301, 9307, 9311, 9315, 9319, 9323, 9335, 9342, 9348, 9355, 9361, 9368, 9374, 9378, 9382, 9386, 9390, 9402, 9409, 9415, 9422, 9428, 9435, 9441, 9445, 9449, 9453, 9457, 9469, 9476, 9482, 9489, 9495, 9502, 9508, 9512, 9516, 9520, 9524, 9536, 9539, 9542, 9545, 9548, 9551, 9554, 9555, 9558]
    sp_jac_run_nia = 697
    sp_jac_run_nja = 697
    return sp_jac_run_ia, sp_jac_run_ja, sp_jac_run_nia, sp_jac_run_nja 

def sp_jac_trap_vectors():

    sp_jac_trap_ia = [0, 695, 1, 31, 32, 37, 38, 351, 352, 353, 354, 2, 33, 34, 37, 38, 351, 352, 353, 354, 3, 35, 36, 37, 38, 351, 352, 353, 354, 4, 103, 104, 109, 110, 375, 376, 377, 378, 5, 105, 106, 109, 110, 375, 376, 377, 378, 6, 107, 108, 109, 110, 375, 376, 377, 378, 7, 135, 136, 141, 142, 395, 396, 397, 398, 8, 137, 138, 141, 142, 395, 396, 397, 398, 9, 139, 140, 141, 142, 395, 396, 397, 398, 10, 183, 184, 189, 190, 407, 408, 409, 410, 11, 185, 186, 189, 190, 407, 408, 409, 410, 12, 187, 188, 189, 190, 407, 408, 409, 410, 13, 191, 192, 197, 198, 411, 412, 413, 414, 14, 193, 194, 197, 198, 411, 412, 413, 414, 15, 195, 196, 197, 198, 411, 412, 413, 414, 16, 255, 256, 261, 262, 427, 428, 429, 430, 17, 257, 258, 261, 262, 427, 428, 429, 430, 18, 259, 260, 261, 262, 427, 428, 429, 430, 19, 271, 272, 277, 278, 431, 432, 433, 434, 20, 273, 274, 277, 278, 431, 432, 433, 434, 21, 275, 276, 277, 278, 431, 432, 433, 434, 22, 311, 312, 317, 318, 439, 440, 441, 442, 23, 313, 314, 317, 318, 439, 440, 441, 442, 24, 315, 316, 317, 318, 439, 440, 441, 442, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 175, 176, 177, 178, 191, 192, 193, 194, 689, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 175, 176, 177, 178, 191, 192, 193, 194, 692, 25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 177, 178, 179, 180, 193, 194, 195, 196, 690, 25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 177, 178, 179, 180, 193, 194, 195, 196, 693, 25, 26, 27, 28, 29, 30, 31, 32, 35, 36, 175, 176, 179, 180, 191, 192, 195, 196, 691, 25, 26, 27, 28, 29, 30, 31, 32, 35, 36, 175, 176, 179, 180, 191, 192, 195, 196, 694, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 455, 601, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 456, 602, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 457, 603, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 458, 604, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 459, 605, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 460, 606, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 461, 607, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 462, 608, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117, 118, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117, 118, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117, 118, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117, 118, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117, 118, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117, 118, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117, 118, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117, 118, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 119, 120, 121, 122, 123, 124, 125, 126, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 119, 120, 121, 122, 123, 124, 125, 126, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 119, 120, 121, 122, 123, 124, 125, 126, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 119, 120, 121, 122, 123, 124, 125, 126, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 119, 120, 121, 122, 123, 124, 125, 126, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 119, 120, 121, 122, 123, 124, 125, 126, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 119, 120, 121, 122, 123, 124, 125, 126, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 119, 120, 121, 122, 123, 124, 125, 126, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 151, 152, 153, 154, 155, 156, 157, 158, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 151, 152, 153, 154, 155, 156, 157, 158, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 151, 152, 153, 154, 155, 156, 157, 158, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 151, 152, 153, 154, 155, 156, 157, 158, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 151, 152, 153, 154, 155, 156, 157, 158, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 151, 152, 153, 154, 155, 156, 157, 158, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 151, 152, 153, 154, 155, 156, 157, 158, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 151, 152, 153, 154, 155, 156, 157, 158, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 159, 160, 161, 162, 163, 164, 165, 166, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 159, 160, 161, 162, 163, 164, 165, 166, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 159, 160, 161, 162, 163, 164, 165, 166, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 159, 160, 161, 162, 163, 164, 165, 166, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 159, 160, 161, 162, 163, 164, 165, 166, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 159, 160, 161, 162, 163, 164, 165, 166, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 159, 160, 161, 162, 163, 164, 165, 166, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 159, 160, 161, 162, 163, 164, 165, 166, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 623, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 624, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 625, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 626, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 627, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 628, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 629, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 630, 47, 48, 49, 50, 51, 52, 53, 54, 111, 112, 113, 114, 115, 116, 117, 118, 463, 47, 48, 49, 50, 51, 52, 53, 54, 111, 112, 113, 114, 115, 116, 117, 118, 464, 47, 48, 49, 50, 51, 52, 53, 54, 111, 112, 113, 114, 115, 116, 117, 118, 465, 47, 48, 49, 50, 51, 52, 53, 54, 111, 112, 113, 114, 115, 116, 117, 118, 466, 47, 48, 49, 50, 51, 52, 53, 54, 111, 112, 113, 114, 115, 116, 117, 118, 467, 47, 48, 49, 50, 51, 52, 53, 54, 111, 112, 113, 114, 115, 116, 117, 118, 468, 47, 48, 49, 50, 51, 52, 53, 54, 111, 112, 113, 114, 115, 116, 117, 118, 469, 47, 48, 49, 50, 51, 52, 53, 54, 111, 112, 113, 114, 115, 116, 117, 118, 470, 55, 56, 57, 58, 59, 60, 61, 62, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 55, 56, 57, 58, 59, 60, 61, 62, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 55, 56, 57, 58, 59, 60, 61, 62, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 55, 56, 57, 58, 59, 60, 61, 62, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 55, 56, 57, 58, 59, 60, 61, 62, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 55, 56, 57, 58, 59, 60, 61, 62, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 55, 56, 57, 58, 59, 60, 61, 62, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 55, 56, 57, 58, 59, 60, 61, 62, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 634, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 635, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 636, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 637, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 638, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 639, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 640, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 641, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 471, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 472, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 473, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 474, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 475, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 476, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 477, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 478, 71, 72, 73, 74, 75, 76, 77, 78, 151, 152, 153, 154, 155, 156, 157, 158, 479, 71, 72, 73, 74, 75, 76, 77, 78, 151, 152, 153, 154, 155, 156, 157, 158, 480, 71, 72, 73, 74, 75, 76, 77, 78, 151, 152, 153, 154, 155, 156, 157, 158, 481, 71, 72, 73, 74, 75, 76, 77, 78, 151, 152, 153, 154, 155, 156, 157, 158, 482, 71, 72, 73, 74, 75, 76, 77, 78, 151, 152, 153, 154, 155, 156, 157, 158, 483, 71, 72, 73, 74, 75, 76, 77, 78, 151, 152, 153, 154, 155, 156, 157, 158, 484, 71, 72, 73, 74, 75, 76, 77, 78, 151, 152, 153, 154, 155, 156, 157, 158, 485, 71, 72, 73, 74, 75, 76, 77, 78, 151, 152, 153, 154, 155, 156, 157, 158, 486, 95, 96, 97, 98, 99, 100, 101, 102, 159, 160, 161, 162, 163, 164, 165, 166, 487, 95, 96, 97, 98, 99, 100, 101, 102, 159, 160, 161, 162, 163, 164, 165, 166, 488, 95, 96, 97, 98, 99, 100, 101, 102, 159, 160, 161, 162, 163, 164, 165, 166, 489, 95, 96, 97, 98, 99, 100, 101, 102, 159, 160, 161, 162, 163, 164, 165, 166, 490, 95, 96, 97, 98, 99, 100, 101, 102, 159, 160, 161, 162, 163, 164, 165, 166, 491, 95, 96, 97, 98, 99, 100, 101, 102, 159, 160, 161, 162, 163, 164, 165, 166, 492, 95, 96, 97, 98, 99, 100, 101, 102, 159, 160, 161, 162, 163, 164, 165, 166, 493, 95, 96, 97, 98, 99, 100, 101, 102, 159, 160, 161, 162, 163, 164, 165, 166, 494, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 495, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 496, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 497, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 498, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 499, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 500, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 501, 103, 104, 105, 106, 107, 108, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174, 502, 25, 26, 29, 30, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 589, 25, 26, 29, 30, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 590, 25, 26, 27, 28, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 591, 25, 26, 27, 28, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 592, 27, 28, 29, 30, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 593, 27, 28, 29, 30, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 594, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 595, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 596, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 503, 645, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 504, 646, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 505, 647, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 506, 648, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 507, 649, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 508, 650, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 509, 651, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 510, 652, 25, 26, 29, 30, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 511, 612, 25, 26, 29, 30, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 512, 613, 25, 26, 27, 28, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 513, 614, 25, 26, 27, 28, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 514, 615, 27, 28, 29, 30, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 515, 616, 27, 28, 29, 30, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 516, 617, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 517, 618, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 518, 619, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 263, 264, 265, 266, 267, 268, 269, 270, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 263, 264, 265, 266, 267, 268, 269, 270, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 263, 264, 265, 266, 267, 268, 269, 270, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 263, 264, 265, 266, 267, 268, 269, 270, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 263, 264, 265, 266, 267, 268, 269, 270, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 263, 264, 265, 266, 267, 268, 269, 270, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 263, 264, 265, 266, 267, 268, 269, 270, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 263, 264, 265, 266, 267, 268, 269, 270, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 303, 304, 305, 306, 307, 308, 309, 310, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 303, 304, 305, 306, 307, 308, 309, 310, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 303, 304, 305, 306, 307, 308, 309, 310, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 303, 304, 305, 306, 307, 308, 309, 310, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 303, 304, 305, 306, 307, 308, 309, 310, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 303, 304, 305, 306, 307, 308, 309, 310, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 303, 304, 305, 306, 307, 308, 309, 310, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 303, 304, 305, 306, 307, 308, 309, 310, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 335, 336, 337, 338, 339, 340, 341, 342, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 335, 336, 337, 338, 339, 340, 341, 342, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 335, 336, 337, 338, 339, 340, 341, 342, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 335, 336, 337, 338, 339, 340, 341, 342, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 335, 336, 337, 338, 339, 340, 341, 342, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 335, 336, 337, 338, 339, 340, 341, 342, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 335, 336, 337, 338, 339, 340, 341, 342, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 335, 336, 337, 338, 339, 340, 341, 342, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 656, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 657, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 658, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 659, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 660, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 661, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 662, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 663, 207, 208, 209, 210, 211, 212, 213, 214, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 295, 296, 297, 298, 299, 300, 301, 302, 207, 208, 209, 210, 211, 212, 213, 214, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 295, 296, 297, 298, 299, 300, 301, 302, 207, 208, 209, 210, 211, 212, 213, 214, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 295, 296, 297, 298, 299, 300, 301, 302, 207, 208, 209, 210, 211, 212, 213, 214, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 295, 296, 297, 298, 299, 300, 301, 302, 207, 208, 209, 210, 211, 212, 213, 214, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 295, 296, 297, 298, 299, 300, 301, 302, 207, 208, 209, 210, 211, 212, 213, 214, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 295, 296, 297, 298, 299, 300, 301, 302, 207, 208, 209, 210, 211, 212, 213, 214, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 295, 296, 297, 298, 299, 300, 301, 302, 207, 208, 209, 210, 211, 212, 213, 214, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 295, 296, 297, 298, 299, 300, 301, 302, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 667, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 668, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 669, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 670, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 671, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 672, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 673, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 674, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 519, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 520, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 521, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 522, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 523, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 524, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 525, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 526, 271, 272, 273, 274, 275, 276, 277, 278, 287, 288, 289, 290, 291, 292, 293, 294, 527, 271, 272, 273, 274, 275, 276, 277, 278, 287, 288, 289, 290, 291, 292, 293, 294, 528, 271, 272, 273, 274, 275, 276, 277, 278, 287, 288, 289, 290, 291, 292, 293, 294, 529, 271, 272, 273, 274, 275, 276, 277, 278, 287, 288, 289, 290, 291, 292, 293, 294, 530, 271, 272, 273, 274, 275, 276, 277, 278, 287, 288, 289, 290, 291, 292, 293, 294, 531, 271, 272, 273, 274, 275, 276, 277, 278, 287, 288, 289, 290, 291, 292, 293, 294, 532, 271, 272, 273, 274, 275, 276, 277, 278, 287, 288, 289, 290, 291, 292, 293, 294, 533, 271, 272, 273, 274, 275, 276, 277, 278, 287, 288, 289, 290, 291, 292, 293, 294, 534, 263, 264, 265, 266, 267, 268, 269, 270, 295, 296, 297, 298, 299, 300, 301, 302, 535, 263, 264, 265, 266, 267, 268, 269, 270, 295, 296, 297, 298, 299, 300, 301, 302, 536, 263, 264, 265, 266, 267, 268, 269, 270, 295, 296, 297, 298, 299, 300, 301, 302, 537, 263, 264, 265, 266, 267, 268, 269, 270, 295, 296, 297, 298, 299, 300, 301, 302, 538, 263, 264, 265, 266, 267, 268, 269, 270, 295, 296, 297, 298, 299, 300, 301, 302, 539, 263, 264, 265, 266, 267, 268, 269, 270, 295, 296, 297, 298, 299, 300, 301, 302, 540, 263, 264, 265, 266, 267, 268, 269, 270, 295, 296, 297, 298, 299, 300, 301, 302, 541, 263, 264, 265, 266, 267, 268, 269, 270, 295, 296, 297, 298, 299, 300, 301, 302, 542, 223, 224, 225, 226, 227, 228, 229, 230, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 327, 328, 329, 330, 331, 332, 333, 334, 223, 224, 225, 226, 227, 228, 229, 230, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 327, 328, 329, 330, 331, 332, 333, 334, 223, 224, 225, 226, 227, 228, 229, 230, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 327, 328, 329, 330, 331, 332, 333, 334, 223, 224, 225, 226, 227, 228, 229, 230, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 327, 328, 329, 330, 331, 332, 333, 334, 223, 224, 225, 226, 227, 228, 229, 230, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 327, 328, 329, 330, 331, 332, 333, 334, 223, 224, 225, 226, 227, 228, 229, 230, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 327, 328, 329, 330, 331, 332, 333, 334, 223, 224, 225, 226, 227, 228, 229, 230, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 327, 328, 329, 330, 331, 332, 333, 334, 223, 224, 225, 226, 227, 228, 229, 230, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 327, 328, 329, 330, 331, 332, 333, 334, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 678, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 679, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 680, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 681, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 682, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 683, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 684, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 685, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 543, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 544, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 545, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 546, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 547, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 548, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 549, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 550, 303, 304, 305, 306, 307, 308, 309, 310, 327, 328, 329, 330, 331, 332, 333, 334, 551, 303, 304, 305, 306, 307, 308, 309, 310, 327, 328, 329, 330, 331, 332, 333, 334, 552, 303, 304, 305, 306, 307, 308, 309, 310, 327, 328, 329, 330, 331, 332, 333, 334, 553, 303, 304, 305, 306, 307, 308, 309, 310, 327, 328, 329, 330, 331, 332, 333, 334, 554, 303, 304, 305, 306, 307, 308, 309, 310, 327, 328, 329, 330, 331, 332, 333, 334, 555, 303, 304, 305, 306, 307, 308, 309, 310, 327, 328, 329, 330, 331, 332, 333, 334, 556, 303, 304, 305, 306, 307, 308, 309, 310, 327, 328, 329, 330, 331, 332, 333, 334, 557, 303, 304, 305, 306, 307, 308, 309, 310, 327, 328, 329, 330, 331, 332, 333, 334, 558, 247, 248, 249, 250, 251, 252, 253, 254, 335, 336, 337, 338, 339, 340, 341, 342, 559, 247, 248, 249, 250, 251, 252, 253, 254, 335, 336, 337, 338, 339, 340, 341, 342, 560, 247, 248, 249, 250, 251, 252, 253, 254, 335, 336, 337, 338, 339, 340, 341, 342, 561, 247, 248, 249, 250, 251, 252, 253, 254, 335, 336, 337, 338, 339, 340, 341, 342, 562, 247, 248, 249, 250, 251, 252, 253, 254, 335, 336, 337, 338, 339, 340, 341, 342, 563, 247, 248, 249, 250, 251, 252, 253, 254, 335, 336, 337, 338, 339, 340, 341, 342, 564, 247, 248, 249, 250, 251, 252, 253, 254, 335, 336, 337, 338, 339, 340, 341, 342, 565, 247, 248, 249, 250, 251, 252, 253, 254, 335, 336, 337, 338, 339, 340, 341, 342, 566, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 567, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 568, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 569, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 570, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 571, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 572, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 573, 255, 256, 257, 258, 259, 260, 261, 262, 343, 344, 345, 346, 347, 348, 349, 350, 574, 351, 355, 609, 352, 356, 353, 357, 610, 354, 358, 351, 355, 359, 379, 352, 356, 360, 380, 353, 357, 361, 381, 354, 358, 362, 382, 355, 359, 363, 395, 356, 360, 364, 396, 357, 361, 365, 397, 358, 362, 366, 398, 359, 363, 367, 383, 360, 364, 368, 384, 361, 365, 369, 385, 362, 366, 370, 386, 363, 367, 371, 407, 364, 368, 372, 408, 365, 369, 373, 409, 366, 370, 374, 410, 367, 371, 375, 387, 368, 372, 376, 388, 369, 373, 377, 389, 370, 374, 378, 390, 371, 375, 391, 631, 372, 376, 392, 373, 377, 393, 632, 374, 378, 394, 355, 379, 576, 356, 380, 357, 381, 576, 358, 382, 363, 383, 577, 364, 384, 365, 385, 577, 366, 386, 371, 387, 578, 372, 388, 373, 389, 578, 374, 390, 375, 391, 579, 376, 392, 377, 393, 579, 378, 394, 359, 395, 399, 642, 360, 396, 400, 361, 397, 401, 643, 362, 398, 402, 395, 399, 575, 396, 400, 397, 401, 575, 398, 402, 403, 407, 597, 404, 408, 405, 409, 598, 406, 410, 367, 403, 407, 447, 580, 653, 368, 404, 408, 448, 369, 405, 409, 449, 580, 654, 370, 406, 410, 450, 411, 415, 620, 412, 416, 413, 417, 621, 414, 418, 411, 415, 419, 431, 412, 416, 420, 432, 413, 417, 421, 433, 414, 418, 422, 434, 415, 419, 423, 439, 416, 420, 424, 440, 417, 421, 425, 441, 418, 422, 426, 442, 419, 423, 427, 447, 420, 424, 428, 448, 421, 425, 429, 449, 422, 426, 430, 450, 423, 427, 451, 664, 424, 428, 452, 425, 429, 453, 665, 426, 430, 454, 415, 431, 435, 675, 416, 432, 436, 417, 433, 437, 676, 418, 434, 438, 431, 435, 581, 432, 436, 433, 437, 581, 434, 438, 419, 439, 443, 686, 420, 440, 444, 421, 441, 445, 687, 422, 442, 446, 439, 443, 582, 440, 444, 441, 445, 582, 442, 446, 407, 423, 447, 583, 408, 424, 448, 409, 425, 449, 583, 410, 426, 450, 427, 451, 584, 428, 452, 429, 453, 584, 430, 454, 31, 32, 37, 38, 455, 456, 33, 34, 37, 38, 457, 458, 35, 36, 37, 38, 459, 460, 31, 32, 37, 38, 455, 456, 33, 34, 37, 38, 457, 458, 35, 36, 37, 38, 459, 460, 455, 457, 459, 461, 456, 458, 460, 462, 111, 112, 117, 118, 463, 464, 113, 114, 117, 118, 465, 466, 115, 116, 117, 118, 467, 468, 111, 112, 117, 118, 463, 464, 113, 114, 117, 118, 465, 466, 115, 116, 117, 118, 467, 468, 463, 465, 467, 469, 464, 466, 468, 470, 143, 144, 149, 150, 471, 472, 145, 146, 149, 150, 473, 474, 147, 148, 149, 150, 475, 476, 143, 144, 149, 150, 471, 472, 145, 146, 149, 150, 473, 474, 147, 148, 149, 150, 475, 476, 471, 473, 475, 477, 472, 474, 476, 478, 151, 152, 157, 158, 479, 480, 153, 154, 157, 158, 481, 482, 155, 156, 157, 158, 483, 484, 151, 152, 157, 158, 479, 480, 153, 154, 157, 158, 481, 482, 155, 156, 157, 158, 483, 484, 479, 481, 483, 485, 480, 482, 484, 486, 159, 160, 165, 166, 487, 488, 161, 162, 165, 166, 489, 490, 163, 164, 165, 166, 491, 492, 159, 160, 165, 166, 487, 488, 161, 162, 165, 166, 489, 490, 163, 164, 165, 166, 491, 492, 487, 489, 491, 493, 488, 490, 492, 494, 167, 168, 173, 174, 495, 496, 169, 170, 173, 174, 497, 498, 171, 172, 173, 174, 499, 500, 167, 168, 173, 174, 495, 496, 169, 170, 173, 174, 497, 498, 171, 172, 173, 174, 499, 500, 495, 497, 499, 501, 496, 498, 500, 502, 183, 184, 189, 190, 503, 504, 185, 186, 189, 190, 505, 506, 187, 188, 189, 190, 507, 508, 183, 184, 189, 190, 503, 504, 185, 186, 189, 190, 505, 506, 187, 188, 189, 190, 507, 508, 503, 505, 507, 509, 504, 506, 508, 510, 191, 192, 197, 198, 511, 512, 193, 194, 197, 198, 513, 514, 195, 196, 197, 198, 515, 516, 191, 192, 197, 198, 511, 512, 193, 194, 197, 198, 513, 514, 195, 196, 197, 198, 515, 516, 511, 513, 515, 517, 512, 514, 516, 518, 279, 280, 285, 286, 519, 520, 281, 282, 285, 286, 521, 522, 283, 284, 285, 286, 523, 524, 279, 280, 285, 286, 519, 520, 281, 282, 285, 286, 521, 522, 283, 284, 285, 286, 523, 524, 519, 521, 523, 525, 520, 522, 524, 526, 287, 288, 293, 294, 527, 528, 289, 290, 293, 294, 529, 530, 291, 292, 293, 294, 531, 532, 287, 288, 293, 294, 527, 528, 289, 290, 293, 294, 529, 530, 291, 292, 293, 294, 531, 532, 527, 529, 531, 533, 528, 530, 532, 534, 295, 296, 301, 302, 535, 536, 297, 298, 301, 302, 537, 538, 299, 300, 301, 302, 539, 540, 295, 296, 301, 302, 535, 536, 297, 298, 301, 302, 537, 538, 299, 300, 301, 302, 539, 540, 535, 537, 539, 541, 536, 538, 540, 542, 319, 320, 325, 326, 543, 544, 321, 322, 325, 326, 545, 546, 323, 324, 325, 326, 547, 548, 319, 320, 325, 326, 543, 544, 321, 322, 325, 326, 545, 546, 323, 324, 325, 326, 547, 548, 543, 545, 547, 549, 544, 546, 548, 550, 327, 328, 333, 334, 551, 552, 329, 330, 333, 334, 553, 554, 331, 332, 333, 334, 555, 556, 327, 328, 333, 334, 551, 552, 329, 330, 333, 334, 553, 554, 331, 332, 333, 334, 555, 556, 551, 553, 555, 557, 552, 554, 556, 558, 335, 336, 341, 342, 559, 560, 337, 338, 341, 342, 561, 562, 339, 340, 341, 342, 563, 564, 335, 336, 341, 342, 559, 560, 337, 338, 341, 342, 561, 562, 339, 340, 341, 342, 563, 564, 559, 561, 563, 565, 560, 562, 564, 566, 343, 344, 349, 350, 567, 568, 345, 346, 349, 350, 569, 570, 347, 348, 349, 350, 571, 572, 343, 344, 349, 350, 567, 568, 345, 346, 349, 350, 569, 570, 347, 348, 349, 350, 571, 572, 567, 569, 571, 573, 568, 570, 572, 574, 399, 401, 575, 379, 381, 576, 383, 385, 577, 387, 389, 578, 391, 393, 579, 407, 409, 580, 435, 437, 581, 443, 445, 582, 447, 449, 583, 451, 453, 584, 585, 600, 586, 600, 587, 600, 181, 182, 588, 595, 596, 175, 176, 181, 182, 585, 589, 590, 595, 596, 175, 176, 181, 182, 589, 590, 177, 178, 181, 182, 586, 591, 592, 595, 596, 177, 178, 181, 182, 591, 592, 179, 180, 181, 182, 587, 593, 594, 595, 596, 179, 180, 181, 182, 593, 594, 589, 591, 593, 595, 590, 592, 594, 596, 403, 597, 599, 405, 598, 599, 597, 598, 599, 597, 598, 600, 1, 31, 32, 37, 38, 601, 602, 31, 32, 37, 38, 601, 602, 2, 33, 34, 37, 38, 603, 604, 33, 34, 37, 38, 603, 604, 3, 35, 36, 37, 38, 605, 606, 35, 36, 37, 38, 605, 606, 601, 603, 605, 607, 602, 604, 606, 608, 351, 353, 609, 611, 351, 353, 610, 611, 1, 2, 3, 601, 602, 603, 604, 605, 606, 607, 608, 611, 13, 191, 192, 197, 198, 612, 613, 191, 192, 197, 198, 612, 613, 14, 193, 194, 197, 198, 614, 615, 193, 194, 197, 198, 614, 615, 15, 195, 196, 197, 198, 616, 617, 195, 196, 197, 198, 616, 617, 612, 614, 616, 618, 613, 615, 617, 619, 411, 413, 620, 622, 411, 413, 621, 622, 13, 14, 15, 612, 613, 614, 615, 616, 617, 618, 619, 622, 4, 103, 104, 109, 110, 623, 624, 103, 104, 109, 110, 623, 624, 5, 105, 106, 109, 110, 625, 626, 105, 106, 109, 110, 625, 626, 6, 107, 108, 109, 110, 627, 628, 107, 108, 109, 110, 627, 628, 623, 625, 627, 629, 624, 626, 628, 630, 375, 377, 631, 633, 375, 377, 632, 633, 4, 5, 6, 623, 624, 625, 626, 627, 628, 629, 630, 633, 7, 135, 136, 141, 142, 634, 635, 135, 136, 141, 142, 634, 635, 8, 137, 138, 141, 142, 636, 637, 137, 138, 141, 142, 636, 637, 9, 139, 140, 141, 142, 638, 639, 139, 140, 141, 142, 638, 639, 634, 636, 638, 640, 635, 637, 639, 641, 395, 397, 642, 644, 395, 397, 643, 644, 7, 8, 9, 634, 635, 636, 637, 638, 639, 640, 641, 644, 10, 183, 184, 189, 190, 645, 646, 183, 184, 189, 190, 645, 646, 11, 185, 186, 189, 190, 647, 648, 185, 186, 189, 190, 647, 648, 12, 187, 188, 189, 190, 649, 650, 187, 188, 189, 190, 649, 650, 645, 647, 649, 651, 646, 648, 650, 652, 407, 409, 653, 655, 407, 409, 654, 655, 10, 11, 12, 645, 646, 647, 648, 649, 650, 651, 652, 655, 16, 255, 256, 261, 262, 656, 657, 255, 256, 261, 262, 656, 657, 17, 257, 258, 261, 262, 658, 659, 257, 258, 261, 262, 658, 659, 18, 259, 260, 261, 262, 660, 661, 259, 260, 261, 262, 660, 661, 656, 658, 660, 662, 657, 659, 661, 663, 427, 429, 664, 666, 427, 429, 665, 666, 16, 17, 18, 656, 657, 658, 659, 660, 661, 662, 663, 666, 19, 271, 272, 277, 278, 667, 668, 271, 272, 277, 278, 667, 668, 20, 273, 274, 277, 278, 669, 670, 273, 274, 277, 278, 669, 670, 21, 275, 276, 277, 278, 671, 672, 275, 276, 277, 278, 671, 672, 667, 669, 671, 673, 668, 670, 672, 674, 431, 433, 675, 677, 431, 433, 676, 677, 19, 20, 21, 667, 668, 669, 670, 671, 672, 673, 674, 677, 22, 311, 312, 317, 318, 678, 679, 311, 312, 317, 318, 678, 679, 23, 313, 314, 317, 318, 680, 681, 313, 314, 317, 318, 680, 681, 24, 315, 316, 317, 318, 682, 683, 315, 316, 317, 318, 682, 683, 678, 680, 682, 684, 679, 681, 683, 685, 439, 441, 686, 688, 439, 441, 687, 688, 22, 23, 24, 678, 679, 680, 681, 682, 683, 684, 685, 688, 25, 689, 692, 27, 690, 693, 29, 691, 694, 26, 689, 692, 28, 690, 693, 30, 691, 694, 695, 0, 695, 696]
    sp_jac_trap_ja = [0, 2, 11, 20, 29, 38, 47, 56, 65, 74, 83, 92, 101, 110, 119, 128, 137, 146, 155, 164, 173, 182, 191, 200, 209, 218, 237, 256, 275, 294, 313, 332, 354, 376, 398, 420, 442, 464, 482, 500, 524, 548, 572, 596, 620, 644, 668, 692, 724, 756, 788, 820, 852, 884, 916, 948, 980, 1012, 1044, 1076, 1108, 1140, 1172, 1204, 1228, 1252, 1276, 1300, 1324, 1348, 1372, 1396, 1428, 1460, 1492, 1524, 1556, 1588, 1620, 1652, 1676, 1700, 1724, 1748, 1772, 1796, 1820, 1844, 1868, 1892, 1916, 1940, 1964, 1988, 2012, 2036, 2068, 2100, 2132, 2164, 2196, 2228, 2260, 2292, 2317, 2342, 2367, 2392, 2417, 2442, 2467, 2492, 2509, 2526, 2543, 2560, 2577, 2594, 2611, 2628, 2652, 2676, 2700, 2724, 2748, 2772, 2796, 2820, 2844, 2868, 2892, 2916, 2940, 2964, 2988, 3012, 3037, 3062, 3087, 3112, 3137, 3162, 3187, 3212, 3229, 3246, 3263, 3280, 3297, 3314, 3331, 3348, 3365, 3382, 3399, 3416, 3433, 3450, 3467, 3484, 3501, 3518, 3535, 3552, 3569, 3586, 3603, 3620, 3637, 3654, 3671, 3688, 3705, 3722, 3739, 3756, 3777, 3798, 3819, 3840, 3861, 3882, 3899, 3916, 3934, 3952, 3970, 3988, 4006, 4024, 4042, 4060, 4082, 4104, 4126, 4148, 4170, 4192, 4210, 4228, 4252, 4276, 4300, 4324, 4348, 4372, 4396, 4420, 4452, 4484, 4516, 4548, 4580, 4612, 4644, 4676, 4700, 4724, 4748, 4772, 4796, 4820, 4844, 4868, 4900, 4932, 4964, 4996, 5028, 5060, 5092, 5124, 5148, 5172, 5196, 5220, 5244, 5268, 5292, 5316, 5340, 5364, 5388, 5412, 5436, 5460, 5484, 5508, 5540, 5572, 5604, 5636, 5668, 5700, 5732, 5764, 5789, 5814, 5839, 5864, 5889, 5914, 5939, 5964, 5996, 6028, 6060, 6092, 6124, 6156, 6188, 6220, 6253, 6286, 6319, 6352, 6385, 6418, 6451, 6484, 6501, 6518, 6535, 6552, 6569, 6586, 6603, 6620, 6637, 6654, 6671, 6688, 6705, 6722, 6739, 6756, 6773, 6790, 6807, 6824, 6841, 6858, 6875, 6892, 6924, 6956, 6988, 7020, 7052, 7084, 7116, 7148, 7173, 7198, 7223, 7248, 7273, 7298, 7323, 7348, 7365, 7382, 7399, 7416, 7433, 7450, 7467, 7484, 7501, 7518, 7535, 7552, 7569, 7586, 7603, 7620, 7637, 7654, 7671, 7688, 7705, 7722, 7739, 7756, 7773, 7790, 7807, 7824, 7841, 7858, 7875, 7892, 7895, 7897, 7900, 7902, 7906, 7910, 7914, 7918, 7922, 7926, 7930, 7934, 7938, 7942, 7946, 7950, 7954, 7958, 7962, 7966, 7970, 7974, 7978, 7982, 7986, 7989, 7993, 7996, 7999, 8001, 8004, 8006, 8009, 8011, 8014, 8016, 8019, 8021, 8024, 8026, 8029, 8031, 8034, 8036, 8040, 8043, 8047, 8050, 8053, 8055, 8058, 8060, 8063, 8065, 8068, 8070, 8076, 8080, 8086, 8090, 8093, 8095, 8098, 8100, 8104, 8108, 8112, 8116, 8120, 8124, 8128, 8132, 8136, 8140, 8144, 8148, 8152, 8155, 8159, 8162, 8166, 8169, 8173, 8176, 8179, 8181, 8184, 8186, 8190, 8193, 8197, 8200, 8203, 8205, 8208, 8210, 8214, 8217, 8221, 8224, 8227, 8229, 8232, 8234, 8240, 8246, 8252, 8258, 8264, 8270, 8274, 8278, 8284, 8290, 8296, 8302, 8308, 8314, 8318, 8322, 8328, 8334, 8340, 8346, 8352, 8358, 8362, 8366, 8372, 8378, 8384, 8390, 8396, 8402, 8406, 8410, 8416, 8422, 8428, 8434, 8440, 8446, 8450, 8454, 8460, 8466, 8472, 8478, 8484, 8490, 8494, 8498, 8504, 8510, 8516, 8522, 8528, 8534, 8538, 8542, 8548, 8554, 8560, 8566, 8572, 8578, 8582, 8586, 8592, 8598, 8604, 8610, 8616, 8622, 8626, 8630, 8636, 8642, 8648, 8654, 8660, 8666, 8670, 8674, 8680, 8686, 8692, 8698, 8704, 8710, 8714, 8718, 8724, 8730, 8736, 8742, 8748, 8754, 8758, 8762, 8768, 8774, 8780, 8786, 8792, 8798, 8802, 8806, 8812, 8818, 8824, 8830, 8836, 8842, 8846, 8850, 8856, 8862, 8868, 8874, 8880, 8886, 8890, 8894, 8897, 8900, 8903, 8906, 8909, 8912, 8915, 8918, 8921, 8924, 8926, 8928, 8930, 8935, 8944, 8950, 8959, 8965, 8974, 8980, 8984, 8988, 8991, 8994, 8997, 9000, 9007, 9013, 9020, 9026, 9033, 9039, 9043, 9047, 9051, 9055, 9067, 9074, 9080, 9087, 9093, 9100, 9106, 9110, 9114, 9118, 9122, 9134, 9141, 9147, 9154, 9160, 9167, 9173, 9177, 9181, 9185, 9189, 9201, 9208, 9214, 9221, 9227, 9234, 9240, 9244, 9248, 9252, 9256, 9268, 9275, 9281, 9288, 9294, 9301, 9307, 9311, 9315, 9319, 9323, 9335, 9342, 9348, 9355, 9361, 9368, 9374, 9378, 9382, 9386, 9390, 9402, 9409, 9415, 9422, 9428, 9435, 9441, 9445, 9449, 9453, 9457, 9469, 9476, 9482, 9489, 9495, 9502, 9508, 9512, 9516, 9520, 9524, 9536, 9539, 9542, 9545, 9548, 9551, 9554, 9555, 9558]
    sp_jac_trap_nia = 697
    sp_jac_trap_nja = 697
    return sp_jac_trap_ia, sp_jac_trap_ja, sp_jac_trap_nia, sp_jac_trap_nja 
