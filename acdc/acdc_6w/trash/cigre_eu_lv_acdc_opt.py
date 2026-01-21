import numpy as np
import numba
import scipy.optimize as sopt
import scipy.sparse as sspa
from scipy.sparse.linalg import spsolve,spilu,splu
from numba import cuda
import cffi
import numba.core.typing.cffi_utils as cffi_support

ffi = cffi.FFI()

import cigre_eu_lv_acdc_opt_cffi as jacs

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

        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 1
        self.N_y = 856 
        self.N_z = 362 
        self.N_store = 100000 
        self.params_list = ['a_R01', 'b_R01', 'c_R01', 'coef_a_R01', 'coef_b_R01', 'coef_c_R01', 'a_R10', 'b_R10', 'c_R10', 'coef_a_R10', 'coef_b_R10', 'coef_c_R10', 'a_R14', 'b_R14', 'c_R14', 'coef_a_R14', 'coef_b_R14', 'coef_c_R14', 'a_I01', 'b_I01', 'c_I01', 'C_a_I01', 'C_b_I01', 'C_c_I01', 'R_dc_H01', 'K_dc_H01', 'a_I02', 'b_I02', 'c_I02', 'coef_a_I02', 'coef_b_I02', 'coef_c_I02', 'a_C01', 'b_C01', 'c_C01', 'coef_a_C01', 'coef_b_C01', 'coef_c_C01', 'a_C09', 'b_C09', 'c_C09', 'coef_a_C09', 'coef_b_C09', 'coef_c_C09', 'a_C11', 'b_C11', 'c_C11', 'coef_a_C11', 'coef_b_C11', 'coef_c_C11', 'a_C16', 'b_C16', 'c_C16', 'coef_a_C16', 'coef_b_C16', 'coef_c_C16'] 
        self.params_values_list  = [2.92, 0.45, 0.027, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 2.92, 0.45, 0.027, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 2.92, 0.45, 0.027, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 2.92, 0.45, 0.027, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 1e-06, 1e-06, 2.92, 0.45, 0.027, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 2.92, 0.45, 0.027, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 2.92, 0.45, 0.027, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 2.92, 0.45, 0.027, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 2.92, 0.45, 0.027, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333] 
        self.inputs_ini_list = ['v_MV0_a_r', 'v_MV0_a_i', 'v_MV0_b_r', 'v_MV0_b_i', 'v_MV0_c_r', 'v_MV0_c_i', 'v_H01_a_r', 'v_H01_a_i', 'v_H01_b_r', 'v_H01_b_i', 'v_H01_c_r', 'v_H01_c_i', 'p_load_R01_a', 'q_load_R01_a', 'p_load_R01_b', 'q_load_R01_b', 'p_load_R01_c', 'q_load_R01_c', 'p_load_R11_a', 'q_load_R11_a', 'p_load_R11_b', 'q_load_R11_b', 'p_load_R11_c', 'q_load_R11_c', 'p_load_R15_a', 'q_load_R15_a', 'p_load_R15_b', 'q_load_R15_b', 'p_load_R15_c', 'q_load_R15_c', 'p_load_R16_a', 'q_load_R16_a', 'p_load_R16_b', 'q_load_R16_b', 'p_load_R16_c', 'q_load_R16_c', 'p_load_R17_a', 'q_load_R17_a', 'p_load_R17_b', 'q_load_R17_b', 'p_load_R17_c', 'q_load_R17_c', 'p_load_R18_a', 'q_load_R18_a', 'p_load_R18_b', 'q_load_R18_b', 'p_load_R18_c', 'q_load_R18_c', 'p_load_I02_a', 'q_load_I02_a', 'p_load_I02_b', 'q_load_I02_b', 'p_load_I02_c', 'q_load_I02_c', 'p_load_C01_a', 'q_load_C01_a', 'p_load_C01_b', 'q_load_C01_b', 'p_load_C01_c', 'q_load_C01_c', 'p_load_C12_a', 'q_load_C12_a', 'p_load_C12_b', 'q_load_C12_b', 'p_load_C12_c', 'q_load_C12_c', 'p_load_C13_a', 'q_load_C13_a', 'p_load_C13_b', 'q_load_C13_b', 'p_load_C13_c', 'q_load_C13_c', 'p_load_C14_a', 'q_load_C14_a', 'p_load_C14_b', 'q_load_C14_b', 'p_load_C14_c', 'q_load_C14_c', 'p_load_C17_a', 'q_load_C17_a', 'p_load_C17_b', 'q_load_C17_b', 'p_load_C17_c', 'q_load_C17_c', 'p_load_C18_a', 'q_load_C18_a', 'p_load_C18_b', 'q_load_C18_b', 'p_load_C18_c', 'q_load_C18_c', 'p_load_C19_a', 'q_load_C19_a', 'p_load_C19_b', 'q_load_C19_b', 'p_load_C19_c', 'q_load_C19_c', 'p_load_C20_a', 'q_load_C20_a', 'p_load_C20_b', 'q_load_C20_b', 'p_load_C20_c', 'q_load_C20_c', 'p_load_S15_a', 'q_load_S15_a', 'p_load_S15_b', 'q_load_S15_b', 'p_load_S15_c', 'q_load_S15_c', 'p_load_S11_a', 'q_load_S11_a', 'p_load_S11_b', 'q_load_S11_b', 'p_load_S11_c', 'q_load_S11_c', 'p_load_S16_a', 'q_load_S16_a', 'p_load_S16_b', 'q_load_S16_b', 'p_load_S16_c', 'q_load_S16_c', 'p_load_S17_a', 'q_load_S17_a', 'p_load_S17_b', 'q_load_S17_b', 'p_load_S17_c', 'q_load_S17_c', 'p_load_S18_a', 'q_load_S18_a', 'p_load_S18_b', 'q_load_S18_b', 'p_load_S18_c', 'q_load_S18_c', 'p_load_H02_a', 'q_load_H02_a', 'p_load_H02_b', 'q_load_H02_b', 'p_load_H02_c', 'q_load_H02_c', 'p_load_D11_a', 'q_load_D11_a', 'p_load_D11_b', 'q_load_D11_b', 'p_load_D11_c', 'q_load_D11_c', 'p_load_D12_a', 'q_load_D12_a', 'p_load_D12_b', 'q_load_D12_b', 'p_load_D12_c', 'q_load_D12_c', 'p_load_D17_a', 'q_load_D17_a', 'p_load_D17_b', 'q_load_D17_b', 'p_load_D17_c', 'q_load_D17_c', 'p_load_D20_a', 'q_load_D20_a', 'p_load_D20_b', 'q_load_D20_b', 'p_load_D20_c', 'q_load_D20_c', 'p_vsc_R01', 'q_vsc_R01', 'p_vsc_R10', 'q_vsc_R10', 'p_vsc_R14', 'q_vsc_R14', 'v_dc_H01_ref', 'q_vsc_I01', 'p_vsc_I02', 'q_vsc_I02', 'p_vsc_C01', 'q_vsc_C01', 'p_vsc_C09', 'q_vsc_C09', 'p_vsc_C11', 'q_vsc_C11', 'p_vsc_C16', 'q_vsc_C16', 'u_dummy'] 
        self.inputs_ini_values_list  = [11547.0, 0.0, -5773.499999999997, -9999.995337498915, -5773.5000000000055, 9999.99533749891, 800.0, 0.0, -0.0004999999999999998, -0.0008660254037844387, -0.0005000000000000004, 0.0008660254037844385, 63333.333333332645, 20816.659994660517, 63333.33333333524, 20816.6599946599, 63333.33333333269, 20816.659994659964, 4750.0, 1561.2494995996008, 4750.000000000066, 1561.249499599505, 4750.000000000209, 1561.2494995995023, 16466.666666666893, 5412.331598611749, 16466.666666667305, 5412.331598612134, 16466.66666666621, 5412.33159861147, 17416.666666666515, 5724.581498533624, 17416.66666666503, 5724.581498529638, 17416.66666666831, 5724.581498531819, 11083.33333333359, 3642.9154990660395, 11083.333333332714, 3642.915499065636, 11083.33333333372, 3642.9154990652505, 14883.333333334202, 4891.915098746447, 14883.333333332934, 4891.915098744666, 14883.333333333358, 4891.91509874458, 28333.333333333823, 17559.42292142097, 28333.33333333306, 17559.422921421217, 28333.33333333302, 17559.422921420693, 36000.000000001164, 17435.59577416271, 36000.000000000626, 17435.59577416234, 35999.99999999973, 17435.595774161826, 5999.999999999965, 2905.9326290272757, 5999.99999999982, 2905.932629026953, 6000.000000000102, 2905.932629027167, 5999.999999999951, 2905.9326290271465, 5999.9999999999345, 2905.9326290269464, 6000.000000000257, 2905.9326290269883, 7500.000000000018, 3632.415786284053, 7499.999999999655, 3632.415786284123, 7500.00000000032, 3632.415786283578, 7500.000000000018, 3632.4157862840566, 7499.999999999873, 3632.4157862842603, 7500.000000000387, 3632.4157862837665, 2399.9999999999377, 1162.3730516107487, 2400.000000000001, 1162.3730516108687, 2399.9999999999077, 1162.3730516109265, 4800.000000000106, 2324.7461032215156, 4800.000000000002, 2324.7461032217375, 4799.999999999984, 2324.746103221428, 2399.999999999949, 1162.3730516108508, 2399.999999999888, 1162.373051610873, 2400.0000000001132, 1162.3730516107257, 192.8323752209726, -9.718819616094102, -20.03722397722062, -0.02905762548866475, -20.037205741971263, -0.029089974871641555, 192.83979553159122, -9.71900796603593, -20.20875722831493, -0.029121096655241097, -20.208738905073265, -0.029152603255041543, 192.85737029008936, -9.719460481505386, -20.641471907102538, -0.02922890152431823, -20.641453366958906, -0.029258077300520324, 192.88281150657224, -9.720111527016815, -21.369191922114958, -0.029277249298687025, -21.369173030512368, -0.029302218689306048, 192.87953299901855, -9.720020740580047, -21.276990644165746, -0.029294919508045192, -21.27697179517636, -0.029320576376925755, 192.9086118391288, -9.720816467721106, -22.18944047816995, -0.029005220727349568, -22.189421218705096, -0.029023895325474225, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 192.84599103120206, -9.71914760286049, -20.37991279670915, -0.02919973118917074, -20.379894384836696, -0.029230627004343868, 192.8611194095459, -9.719538600086846, -20.763055006637444, -0.029276902189468768, -20.7630364044605, -0.029305685082155475, 192.88278168535493, -9.720110069357233, -21.369451373913144, -0.029277159180983725, -21.369432482106635, -0.029302131747405924, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0] 
        self.inputs_run_list = ['v_MV0_a_r', 'v_MV0_a_i', 'v_MV0_b_r', 'v_MV0_b_i', 'v_MV0_c_r', 'v_MV0_c_i', 'v_H01_a_r', 'v_H01_a_i', 'v_H01_b_r', 'v_H01_b_i', 'v_H01_c_r', 'v_H01_c_i', 'p_load_R01_a', 'q_load_R01_a', 'p_load_R01_b', 'q_load_R01_b', 'p_load_R01_c', 'q_load_R01_c', 'p_load_R11_a', 'q_load_R11_a', 'p_load_R11_b', 'q_load_R11_b', 'p_load_R11_c', 'q_load_R11_c', 'p_load_R15_a', 'q_load_R15_a', 'p_load_R15_b', 'q_load_R15_b', 'p_load_R15_c', 'q_load_R15_c', 'p_load_R16_a', 'q_load_R16_a', 'p_load_R16_b', 'q_load_R16_b', 'p_load_R16_c', 'q_load_R16_c', 'p_load_R17_a', 'q_load_R17_a', 'p_load_R17_b', 'q_load_R17_b', 'p_load_R17_c', 'q_load_R17_c', 'p_load_R18_a', 'q_load_R18_a', 'p_load_R18_b', 'q_load_R18_b', 'p_load_R18_c', 'q_load_R18_c', 'p_load_I02_a', 'q_load_I02_a', 'p_load_I02_b', 'q_load_I02_b', 'p_load_I02_c', 'q_load_I02_c', 'p_load_C01_a', 'q_load_C01_a', 'p_load_C01_b', 'q_load_C01_b', 'p_load_C01_c', 'q_load_C01_c', 'p_load_C12_a', 'q_load_C12_a', 'p_load_C12_b', 'q_load_C12_b', 'p_load_C12_c', 'q_load_C12_c', 'p_load_C13_a', 'q_load_C13_a', 'p_load_C13_b', 'q_load_C13_b', 'p_load_C13_c', 'q_load_C13_c', 'p_load_C14_a', 'q_load_C14_a', 'p_load_C14_b', 'q_load_C14_b', 'p_load_C14_c', 'q_load_C14_c', 'p_load_C17_a', 'q_load_C17_a', 'p_load_C17_b', 'q_load_C17_b', 'p_load_C17_c', 'q_load_C17_c', 'p_load_C18_a', 'q_load_C18_a', 'p_load_C18_b', 'q_load_C18_b', 'p_load_C18_c', 'q_load_C18_c', 'p_load_C19_a', 'q_load_C19_a', 'p_load_C19_b', 'q_load_C19_b', 'p_load_C19_c', 'q_load_C19_c', 'p_load_C20_a', 'q_load_C20_a', 'p_load_C20_b', 'q_load_C20_b', 'p_load_C20_c', 'q_load_C20_c', 'p_load_S15_a', 'q_load_S15_a', 'p_load_S15_b', 'q_load_S15_b', 'p_load_S15_c', 'q_load_S15_c', 'p_load_S11_a', 'q_load_S11_a', 'p_load_S11_b', 'q_load_S11_b', 'p_load_S11_c', 'q_load_S11_c', 'p_load_S16_a', 'q_load_S16_a', 'p_load_S16_b', 'q_load_S16_b', 'p_load_S16_c', 'q_load_S16_c', 'p_load_S17_a', 'q_load_S17_a', 'p_load_S17_b', 'q_load_S17_b', 'p_load_S17_c', 'q_load_S17_c', 'p_load_S18_a', 'q_load_S18_a', 'p_load_S18_b', 'q_load_S18_b', 'p_load_S18_c', 'q_load_S18_c', 'p_load_H02_a', 'q_load_H02_a', 'p_load_H02_b', 'q_load_H02_b', 'p_load_H02_c', 'q_load_H02_c', 'p_load_D11_a', 'q_load_D11_a', 'p_load_D11_b', 'q_load_D11_b', 'p_load_D11_c', 'q_load_D11_c', 'p_load_D12_a', 'q_load_D12_a', 'p_load_D12_b', 'q_load_D12_b', 'p_load_D12_c', 'q_load_D12_c', 'p_load_D17_a', 'q_load_D17_a', 'p_load_D17_b', 'q_load_D17_b', 'p_load_D17_c', 'q_load_D17_c', 'p_load_D20_a', 'q_load_D20_a', 'p_load_D20_b', 'q_load_D20_b', 'p_load_D20_c', 'q_load_D20_c', 'p_vsc_R01', 'q_vsc_R01', 'p_vsc_R10', 'q_vsc_R10', 'p_vsc_R14', 'q_vsc_R14', 'v_dc_H01_ref', 'q_vsc_I01', 'p_vsc_I02', 'q_vsc_I02', 'p_vsc_C01', 'q_vsc_C01', 'p_vsc_C09', 'q_vsc_C09', 'p_vsc_C11', 'q_vsc_C11', 'p_vsc_C16', 'q_vsc_C16', 'u_dummy'] 
        self.inputs_run_values_list = [11547.0, 0.0, -5773.499999999997, -9999.995337498915, -5773.5000000000055, 9999.99533749891, 800.0, 0.0, -0.0004999999999999998, -0.0008660254037844387, -0.0005000000000000004, 0.0008660254037844385, 63333.333333332645, 20816.659994660517, 63333.33333333524, 20816.6599946599, 63333.33333333269, 20816.659994659964, 4750.0, 1561.2494995996008, 4750.000000000066, 1561.249499599505, 4750.000000000209, 1561.2494995995023, 16466.666666666893, 5412.331598611749, 16466.666666667305, 5412.331598612134, 16466.66666666621, 5412.33159861147, 17416.666666666515, 5724.581498533624, 17416.66666666503, 5724.581498529638, 17416.66666666831, 5724.581498531819, 11083.33333333359, 3642.9154990660395, 11083.333333332714, 3642.915499065636, 11083.33333333372, 3642.9154990652505, 14883.333333334202, 4891.915098746447, 14883.333333332934, 4891.915098744666, 14883.333333333358, 4891.91509874458, 28333.333333333823, 17559.42292142097, 28333.33333333306, 17559.422921421217, 28333.33333333302, 17559.422921420693, 36000.000000001164, 17435.59577416271, 36000.000000000626, 17435.59577416234, 35999.99999999973, 17435.595774161826, 5999.999999999965, 2905.9326290272757, 5999.99999999982, 2905.932629026953, 6000.000000000102, 2905.932629027167, 5999.999999999951, 2905.9326290271465, 5999.9999999999345, 2905.9326290269464, 6000.000000000257, 2905.9326290269883, 7500.000000000018, 3632.415786284053, 7499.999999999655, 3632.415786284123, 7500.00000000032, 3632.415786283578, 7500.000000000018, 3632.4157862840566, 7499.999999999873, 3632.4157862842603, 7500.000000000387, 3632.4157862837665, 2399.9999999999377, 1162.3730516107487, 2400.000000000001, 1162.3730516108687, 2399.9999999999077, 1162.3730516109265, 4800.000000000106, 2324.7461032215156, 4800.000000000002, 2324.7461032217375, 4799.999999999984, 2324.746103221428, 2399.999999999949, 1162.3730516108508, 2399.999999999888, 1162.373051610873, 2400.0000000001132, 1162.3730516107257, 192.8323752209726, -9.718819616094102, -20.03722397722062, -0.02905762548866475, -20.037205741971263, -0.029089974871641555, 192.83979553159122, -9.71900796603593, -20.20875722831493, -0.029121096655241097, -20.208738905073265, -0.029152603255041543, 192.85737029008936, -9.719460481505386, -20.641471907102538, -0.02922890152431823, -20.641453366958906, -0.029258077300520324, 192.88281150657224, -9.720111527016815, -21.369191922114958, -0.029277249298687025, -21.369173030512368, -0.029302218689306048, 192.87953299901855, -9.720020740580047, -21.276990644165746, -0.029294919508045192, -21.27697179517636, -0.029320576376925755, 192.9086118391288, -9.720816467721106, -22.18944047816995, -0.029005220727349568, -22.189421218705096, -0.029023895325474225, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 192.84599103120206, -9.71914760286049, -20.37991279670915, -0.02919973118917074, -20.379894384836696, -0.029230627004343868, 192.8611194095459, -9.719538600086846, -20.763055006637444, -0.029276902189468768, -20.7630364044605, -0.029305685082155475, 192.88278168535493, -9.720110069357233, -21.369451373913144, -0.029277159180983725, -21.369432482106635, -0.029302131747405924, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0] 
        self.outputs_list = ['i_t_MV0_R01_1_a_r', 'i_t_MV0_R01_1_a_i', 'i_t_MV0_R01_1_b_r', 'i_t_MV0_R01_1_b_i', 'i_t_MV0_R01_1_c_r', 'i_t_MV0_R01_1_c_i', 'p_t_MV0_R01_1', 'q_t_MV0_R01_1', 'i_t_MV0_R01_2_a_r', 'i_t_MV0_R01_2_a_i', 'i_t_MV0_R01_2_b_r', 'i_t_MV0_R01_2_b_i', 'i_t_MV0_R01_2_c_r', 'i_t_MV0_R01_2_c_i', 'i_t_MV0_R01_2_n_r', 'i_t_MV0_R01_2_n_i', 'i_t_MV0_I01_1_a_r', 'i_t_MV0_I01_1_a_i', 'i_t_MV0_I01_1_b_r', 'i_t_MV0_I01_1_b_i', 'i_t_MV0_I01_1_c_r', 'i_t_MV0_I01_1_c_i', 'p_t_MV0_I01_1', 'q_t_MV0_I01_1', 'i_t_MV0_I01_2_a_r', 'i_t_MV0_I01_2_a_i', 'i_t_MV0_I01_2_b_r', 'i_t_MV0_I01_2_b_i', 'i_t_MV0_I01_2_c_r', 'i_t_MV0_I01_2_c_i', 'i_t_MV0_I01_2_n_r', 'i_t_MV0_I01_2_n_i', 'i_t_MV0_C01_1_a_r', 'i_t_MV0_C01_1_a_i', 'i_t_MV0_C01_1_b_r', 'i_t_MV0_C01_1_b_i', 'i_t_MV0_C01_1_c_r', 'i_t_MV0_C01_1_c_i', 'p_t_MV0_C01_1', 'q_t_MV0_C01_1', 'i_t_MV0_C01_2_a_r', 'i_t_MV0_C01_2_a_i', 'i_t_MV0_C01_2_b_r', 'i_t_MV0_C01_2_b_i', 'i_t_MV0_C01_2_c_r', 'i_t_MV0_C01_2_c_i', 'i_t_MV0_C01_2_n_r', 'i_t_MV0_C01_2_n_i', 'i_l_S01_S03_a_r', 'i_l_S01_S03_a_i', 'i_l_S01_S03_b_r', 'i_l_S01_S03_b_i', 'i_l_S01_S03_c_r', 'i_l_S01_S03_c_i', 'i_l_S01_S03_n_r', 'i_l_S01_S03_n_i', 'i_l_H01_H02_a_r', 'i_l_H01_H02_a_i', 'i_l_H01_H02_b_r', 'i_l_H01_H02_b_i', 'i_l_H01_H02_c_r', 'i_l_H01_H02_c_i', 'i_l_H01_H02_n_r', 'i_l_H01_H02_n_i', 'i_l_D01_D03_a_r', 'i_l_D01_D03_a_i', 'i_l_D01_D03_b_r', 'i_l_D01_D03_b_i', 'i_l_D01_D03_c_r', 'i_l_D01_D03_c_i', 'i_l_D01_D03_n_r', 'i_l_D01_D03_n_i', 'p_vsc_R01', 'p_vsc_loss_R01', 'p_vsc_R10', 'p_vsc_loss_R10', 'p_vsc_R14', 'p_vsc_loss_R14', 'p_vsc_I01', 'p_vsc_loss_I01', 'p_vsc_I02', 'p_vsc_loss_I02', 'p_vsc_C01', 'p_vsc_loss_C01', 'p_vsc_C09', 'p_vsc_loss_C09', 'p_vsc_C11', 'p_vsc_loss_C11', 'p_vsc_C16', 'p_vsc_loss_C16', 'v_MV0_a_m', 'v_MV0_b_m', 'v_MV0_c_m', 'v_H01_a_m', 'v_H01_b_m', 'v_H01_c_m', 'v_R01_a_m', 'v_R01_b_m', 'v_R01_c_m', 'v_R01_n_m', 'v_R11_a_m', 'v_R11_b_m', 'v_R11_c_m', 'v_R11_n_m', 'v_R15_a_m', 'v_R15_b_m', 'v_R15_c_m', 'v_R15_n_m', 'v_R16_a_m', 'v_R16_b_m', 'v_R16_c_m', 'v_R16_n_m', 'v_R17_a_m', 'v_R17_b_m', 'v_R17_c_m', 'v_R17_n_m', 'v_R18_a_m', 'v_R18_b_m', 'v_R18_c_m', 'v_R18_n_m', 'v_I02_a_m', 'v_I02_b_m', 'v_I02_c_m', 'v_I02_n_m', 'v_C01_a_m', 'v_C01_b_m', 'v_C01_c_m', 'v_C01_n_m', 'v_C12_a_m', 'v_C12_b_m', 'v_C12_c_m', 'v_C12_n_m', 'v_C13_a_m', 'v_C13_b_m', 'v_C13_c_m', 'v_C13_n_m', 'v_C14_a_m', 'v_C14_b_m', 'v_C14_c_m', 'v_C14_n_m', 'v_C17_a_m', 'v_C17_b_m', 'v_C17_c_m', 'v_C17_n_m', 'v_C18_a_m', 'v_C18_b_m', 'v_C18_c_m', 'v_C18_n_m', 'v_C19_a_m', 'v_C19_b_m', 'v_C19_c_m', 'v_C19_n_m', 'v_C20_a_m', 'v_C20_b_m', 'v_C20_c_m', 'v_C20_n_m', 'v_S15_a_m', 'v_S15_b_m', 'v_S15_c_m', 'v_S15_n_m', 'v_S11_a_m', 'v_S11_b_m', 'v_S11_c_m', 'v_S11_n_m', 'v_S16_a_m', 'v_S16_b_m', 'v_S16_c_m', 'v_S16_n_m', 'v_S17_a_m', 'v_S17_b_m', 'v_S17_c_m', 'v_S17_n_m', 'v_S18_a_m', 'v_S18_b_m', 'v_S18_c_m', 'v_S18_n_m', 'v_H02_a_m', 'v_H02_b_m', 'v_H02_c_m', 'v_H02_n_m', 'v_D11_a_m', 'v_D11_b_m', 'v_D11_c_m', 'v_D11_n_m', 'v_D12_a_m', 'v_D12_b_m', 'v_D12_c_m', 'v_D12_n_m', 'v_D17_a_m', 'v_D17_b_m', 'v_D17_c_m', 'v_D17_n_m', 'v_D20_a_m', 'v_D20_b_m', 'v_D20_c_m', 'v_D20_n_m', 'v_I01_a_m', 'v_I01_b_m', 'v_I01_c_m', 'v_I01_n_m', 'v_R02_a_m', 'v_R02_b_m', 'v_R02_c_m', 'v_R02_n_m', 'v_R03_a_m', 'v_R03_b_m', 'v_R03_c_m', 'v_R03_n_m', 'v_R04_a_m', 'v_R04_b_m', 'v_R04_c_m', 'v_R04_n_m', 'v_R05_a_m', 'v_R05_b_m', 'v_R05_c_m', 'v_R05_n_m', 'v_R06_a_m', 'v_R06_b_m', 'v_R06_c_m', 'v_R06_n_m', 'v_R07_a_m', 'v_R07_b_m', 'v_R07_c_m', 'v_R07_n_m', 'v_R08_a_m', 'v_R08_b_m', 'v_R08_c_m', 'v_R08_n_m', 'v_R09_a_m', 'v_R09_b_m', 'v_R09_c_m', 'v_R09_n_m', 'v_R10_a_m', 'v_R10_b_m', 'v_R10_c_m', 'v_R10_n_m', 'v_R12_a_m', 'v_R12_b_m', 'v_R12_c_m', 'v_R12_n_m', 'v_R13_a_m', 'v_R13_b_m', 'v_R13_c_m', 'v_R13_n_m', 'v_R14_a_m', 'v_R14_b_m', 'v_R14_c_m', 'v_R14_n_m', 'v_C02_a_m', 'v_C02_b_m', 'v_C02_c_m', 'v_C02_n_m', 'v_C03_a_m', 'v_C03_b_m', 'v_C03_c_m', 'v_C03_n_m', 'v_C04_a_m', 'v_C04_b_m', 'v_C04_c_m', 'v_C04_n_m', 'v_C05_a_m', 'v_C05_b_m', 'v_C05_c_m', 'v_C05_n_m', 'v_C06_a_m', 'v_C06_b_m', 'v_C06_c_m', 'v_C06_n_m', 'v_C07_a_m', 'v_C07_b_m', 'v_C07_c_m', 'v_C07_n_m', 'v_C08_a_m', 'v_C08_b_m', 'v_C08_c_m', 'v_C08_n_m', 'v_C09_a_m', 'v_C09_b_m', 'v_C09_c_m', 'v_C09_n_m', 'v_C10_a_m', 'v_C10_b_m', 'v_C10_c_m', 'v_C10_n_m', 'v_C11_a_m', 'v_C11_b_m', 'v_C11_c_m', 'v_C11_n_m', 'v_C15_a_m', 'v_C15_b_m', 'v_C15_c_m', 'v_C15_n_m', 'v_C16_a_m', 'v_C16_b_m', 'v_C16_c_m', 'v_C16_n_m', 'v_S01_a_m', 'v_S01_b_m', 'v_S01_c_m', 'v_S01_n_m', 'v_S03_a_m', 'v_S03_b_m', 'v_S03_c_m', 'v_S03_n_m', 'v_S04_a_m', 'v_S04_b_m', 'v_S04_c_m', 'v_S04_n_m', 'v_S06_a_m', 'v_S06_b_m', 'v_S06_c_m', 'v_S06_n_m', 'v_S07_a_m', 'v_S07_b_m', 'v_S07_c_m', 'v_S07_n_m', 'v_S09_a_m', 'v_S09_b_m', 'v_S09_c_m', 'v_S09_n_m', 'v_S10_a_m', 'v_S10_b_m', 'v_S10_c_m', 'v_S10_n_m', 'v_S14_a_m', 'v_S14_b_m', 'v_S14_c_m', 'v_S14_n_m', 'v_H01_n_m', 'v_D01_a_m', 'v_D01_b_m', 'v_D01_c_m', 'v_D01_n_m', 'v_D03_a_m', 'v_D03_b_m', 'v_D03_c_m', 'v_D03_n_m', 'v_D05_a_m', 'v_D05_b_m', 'v_D05_c_m', 'v_D05_n_m', 'v_D08_a_m', 'v_D08_b_m', 'v_D08_c_m', 'v_D08_n_m', 'v_D09_a_m', 'v_D09_b_m', 'v_D09_c_m', 'v_D09_n_m', 'v_D16_a_m', 'v_D16_b_m', 'v_D16_c_m', 'v_D16_n_m', 'v_D19_a_m', 'v_D19_b_m', 'v_D19_c_m', 'v_D19_n_m', 'p_total', 'i_res', 'i_ind', 'i_com', 'z_dummy'] 
        self.x_list = ['x_dummy'] 
        self.y_run_list = ['v_R01_a_r', 'v_R01_a_i', 'v_R01_b_r', 'v_R01_b_i', 'v_R01_c_r', 'v_R01_c_i', 'v_R01_n_r', 'v_R01_n_i', 'v_R11_a_r', 'v_R11_a_i', 'v_R11_b_r', 'v_R11_b_i', 'v_R11_c_r', 'v_R11_c_i', 'v_R11_n_r', 'v_R11_n_i', 'v_R15_a_r', 'v_R15_a_i', 'v_R15_b_r', 'v_R15_b_i', 'v_R15_c_r', 'v_R15_c_i', 'v_R15_n_r', 'v_R15_n_i', 'v_R16_a_r', 'v_R16_a_i', 'v_R16_b_r', 'v_R16_b_i', 'v_R16_c_r', 'v_R16_c_i', 'v_R16_n_r', 'v_R16_n_i', 'v_R17_a_r', 'v_R17_a_i', 'v_R17_b_r', 'v_R17_b_i', 'v_R17_c_r', 'v_R17_c_i', 'v_R17_n_r', 'v_R17_n_i', 'v_R18_a_r', 'v_R18_a_i', 'v_R18_b_r', 'v_R18_b_i', 'v_R18_c_r', 'v_R18_c_i', 'v_R18_n_r', 'v_R18_n_i', 'v_I02_a_r', 'v_I02_a_i', 'v_I02_b_r', 'v_I02_b_i', 'v_I02_c_r', 'v_I02_c_i', 'v_I02_n_r', 'v_I02_n_i', 'v_C01_a_r', 'v_C01_a_i', 'v_C01_b_r', 'v_C01_b_i', 'v_C01_c_r', 'v_C01_c_i', 'v_C01_n_r', 'v_C01_n_i', 'v_C12_a_r', 'v_C12_a_i', 'v_C12_b_r', 'v_C12_b_i', 'v_C12_c_r', 'v_C12_c_i', 'v_C12_n_r', 'v_C12_n_i', 'v_C13_a_r', 'v_C13_a_i', 'v_C13_b_r', 'v_C13_b_i', 'v_C13_c_r', 'v_C13_c_i', 'v_C13_n_r', 'v_C13_n_i', 'v_C14_a_r', 'v_C14_a_i', 'v_C14_b_r', 'v_C14_b_i', 'v_C14_c_r', 'v_C14_c_i', 'v_C14_n_r', 'v_C14_n_i', 'v_C17_a_r', 'v_C17_a_i', 'v_C17_b_r', 'v_C17_b_i', 'v_C17_c_r', 'v_C17_c_i', 'v_C17_n_r', 'v_C17_n_i', 'v_C18_a_r', 'v_C18_a_i', 'v_C18_b_r', 'v_C18_b_i', 'v_C18_c_r', 'v_C18_c_i', 'v_C18_n_r', 'v_C18_n_i', 'v_C19_a_r', 'v_C19_a_i', 'v_C19_b_r', 'v_C19_b_i', 'v_C19_c_r', 'v_C19_c_i', 'v_C19_n_r', 'v_C19_n_i', 'v_C20_a_r', 'v_C20_a_i', 'v_C20_b_r', 'v_C20_b_i', 'v_C20_c_r', 'v_C20_c_i', 'v_C20_n_r', 'v_C20_n_i', 'v_S15_a_r', 'v_S15_a_i', 'v_S15_b_r', 'v_S15_b_i', 'v_S15_c_r', 'v_S15_c_i', 'v_S15_n_r', 'v_S15_n_i', 'v_S11_a_r', 'v_S11_a_i', 'v_S11_b_r', 'v_S11_b_i', 'v_S11_c_r', 'v_S11_c_i', 'v_S11_n_r', 'v_S11_n_i', 'v_S16_a_r', 'v_S16_a_i', 'v_S16_b_r', 'v_S16_b_i', 'v_S16_c_r', 'v_S16_c_i', 'v_S16_n_r', 'v_S16_n_i', 'v_S17_a_r', 'v_S17_a_i', 'v_S17_b_r', 'v_S17_b_i', 'v_S17_c_r', 'v_S17_c_i', 'v_S17_n_r', 'v_S17_n_i', 'v_S18_a_r', 'v_S18_a_i', 'v_S18_b_r', 'v_S18_b_i', 'v_S18_c_r', 'v_S18_c_i', 'v_S18_n_r', 'v_S18_n_i', 'v_H02_a_r', 'v_H02_a_i', 'v_H02_b_r', 'v_H02_b_i', 'v_H02_c_r', 'v_H02_c_i', 'v_H02_n_r', 'v_H02_n_i', 'v_D11_a_r', 'v_D11_a_i', 'v_D11_b_r', 'v_D11_b_i', 'v_D11_c_r', 'v_D11_c_i', 'v_D11_n_r', 'v_D11_n_i', 'v_D12_a_r', 'v_D12_a_i', 'v_D12_b_r', 'v_D12_b_i', 'v_D12_c_r', 'v_D12_c_i', 'v_D12_n_r', 'v_D12_n_i', 'v_D17_a_r', 'v_D17_a_i', 'v_D17_b_r', 'v_D17_b_i', 'v_D17_c_r', 'v_D17_c_i', 'v_D17_n_r', 'v_D17_n_i', 'v_D20_a_r', 'v_D20_a_i', 'v_D20_b_r', 'v_D20_b_i', 'v_D20_c_r', 'v_D20_c_i', 'v_D20_n_r', 'v_D20_n_i', 'v_I01_a_r', 'v_I01_a_i', 'v_I01_b_r', 'v_I01_b_i', 'v_I01_c_r', 'v_I01_c_i', 'v_I01_n_r', 'v_I01_n_i', 'v_R02_a_r', 'v_R02_a_i', 'v_R02_b_r', 'v_R02_b_i', 'v_R02_c_r', 'v_R02_c_i', 'v_R02_n_r', 'v_R02_n_i', 'v_R03_a_r', 'v_R03_a_i', 'v_R03_b_r', 'v_R03_b_i', 'v_R03_c_r', 'v_R03_c_i', 'v_R03_n_r', 'v_R03_n_i', 'v_R04_a_r', 'v_R04_a_i', 'v_R04_b_r', 'v_R04_b_i', 'v_R04_c_r', 'v_R04_c_i', 'v_R04_n_r', 'v_R04_n_i', 'v_R05_a_r', 'v_R05_a_i', 'v_R05_b_r', 'v_R05_b_i', 'v_R05_c_r', 'v_R05_c_i', 'v_R05_n_r', 'v_R05_n_i', 'v_R06_a_r', 'v_R06_a_i', 'v_R06_b_r', 'v_R06_b_i', 'v_R06_c_r', 'v_R06_c_i', 'v_R06_n_r', 'v_R06_n_i', 'v_R07_a_r', 'v_R07_a_i', 'v_R07_b_r', 'v_R07_b_i', 'v_R07_c_r', 'v_R07_c_i', 'v_R07_n_r', 'v_R07_n_i', 'v_R08_a_r', 'v_R08_a_i', 'v_R08_b_r', 'v_R08_b_i', 'v_R08_c_r', 'v_R08_c_i', 'v_R08_n_r', 'v_R08_n_i', 'v_R09_a_r', 'v_R09_a_i', 'v_R09_b_r', 'v_R09_b_i', 'v_R09_c_r', 'v_R09_c_i', 'v_R09_n_r', 'v_R09_n_i', 'v_R10_a_r', 'v_R10_a_i', 'v_R10_b_r', 'v_R10_b_i', 'v_R10_c_r', 'v_R10_c_i', 'v_R10_n_r', 'v_R10_n_i', 'v_R12_a_r', 'v_R12_a_i', 'v_R12_b_r', 'v_R12_b_i', 'v_R12_c_r', 'v_R12_c_i', 'v_R12_n_r', 'v_R12_n_i', 'v_R13_a_r', 'v_R13_a_i', 'v_R13_b_r', 'v_R13_b_i', 'v_R13_c_r', 'v_R13_c_i', 'v_R13_n_r', 'v_R13_n_i', 'v_R14_a_r', 'v_R14_a_i', 'v_R14_b_r', 'v_R14_b_i', 'v_R14_c_r', 'v_R14_c_i', 'v_R14_n_r', 'v_R14_n_i', 'v_C02_a_r', 'v_C02_a_i', 'v_C02_b_r', 'v_C02_b_i', 'v_C02_c_r', 'v_C02_c_i', 'v_C02_n_r', 'v_C02_n_i', 'v_C03_a_r', 'v_C03_a_i', 'v_C03_b_r', 'v_C03_b_i', 'v_C03_c_r', 'v_C03_c_i', 'v_C03_n_r', 'v_C03_n_i', 'v_C04_a_r', 'v_C04_a_i', 'v_C04_b_r', 'v_C04_b_i', 'v_C04_c_r', 'v_C04_c_i', 'v_C04_n_r', 'v_C04_n_i', 'v_C05_a_r', 'v_C05_a_i', 'v_C05_b_r', 'v_C05_b_i', 'v_C05_c_r', 'v_C05_c_i', 'v_C05_n_r', 'v_C05_n_i', 'v_C06_a_r', 'v_C06_a_i', 'v_C06_b_r', 'v_C06_b_i', 'v_C06_c_r', 'v_C06_c_i', 'v_C06_n_r', 'v_C06_n_i', 'v_C07_a_r', 'v_C07_a_i', 'v_C07_b_r', 'v_C07_b_i', 'v_C07_c_r', 'v_C07_c_i', 'v_C07_n_r', 'v_C07_n_i', 'v_C08_a_r', 'v_C08_a_i', 'v_C08_b_r', 'v_C08_b_i', 'v_C08_c_r', 'v_C08_c_i', 'v_C08_n_r', 'v_C08_n_i', 'v_C09_a_r', 'v_C09_a_i', 'v_C09_b_r', 'v_C09_b_i', 'v_C09_c_r', 'v_C09_c_i', 'v_C09_n_r', 'v_C09_n_i', 'v_C10_a_r', 'v_C10_a_i', 'v_C10_b_r', 'v_C10_b_i', 'v_C10_c_r', 'v_C10_c_i', 'v_C10_n_r', 'v_C10_n_i', 'v_C11_a_r', 'v_C11_a_i', 'v_C11_b_r', 'v_C11_b_i', 'v_C11_c_r', 'v_C11_c_i', 'v_C11_n_r', 'v_C11_n_i', 'v_C15_a_r', 'v_C15_a_i', 'v_C15_b_r', 'v_C15_b_i', 'v_C15_c_r', 'v_C15_c_i', 'v_C15_n_r', 'v_C15_n_i', 'v_C16_a_r', 'v_C16_a_i', 'v_C16_b_r', 'v_C16_b_i', 'v_C16_c_r', 'v_C16_c_i', 'v_C16_n_r', 'v_C16_n_i', 'v_S01_a_r', 'v_S01_a_i', 'v_S01_b_r', 'v_S01_b_i', 'v_S01_c_r', 'v_S01_c_i', 'v_S01_n_r', 'v_S01_n_i', 'v_S03_a_r', 'v_S03_a_i', 'v_S03_b_r', 'v_S03_b_i', 'v_S03_c_r', 'v_S03_c_i', 'v_S03_n_r', 'v_S03_n_i', 'v_S04_a_r', 'v_S04_a_i', 'v_S04_b_r', 'v_S04_b_i', 'v_S04_c_r', 'v_S04_c_i', 'v_S04_n_r', 'v_S04_n_i', 'v_S06_a_r', 'v_S06_a_i', 'v_S06_b_r', 'v_S06_b_i', 'v_S06_c_r', 'v_S06_c_i', 'v_S06_n_r', 'v_S06_n_i', 'v_S07_a_r', 'v_S07_a_i', 'v_S07_b_r', 'v_S07_b_i', 'v_S07_c_r', 'v_S07_c_i', 'v_S07_n_r', 'v_S07_n_i', 'v_S09_a_r', 'v_S09_a_i', 'v_S09_b_r', 'v_S09_b_i', 'v_S09_c_r', 'v_S09_c_i', 'v_S09_n_r', 'v_S09_n_i', 'v_S10_a_r', 'v_S10_a_i', 'v_S10_b_r', 'v_S10_b_i', 'v_S10_c_r', 'v_S10_c_i', 'v_S10_n_r', 'v_S10_n_i', 'v_S14_a_r', 'v_S14_a_i', 'v_S14_b_r', 'v_S14_b_i', 'v_S14_c_r', 'v_S14_c_i', 'v_S14_n_r', 'v_S14_n_i', 'v_H01_n_r', 'v_H01_n_i', 'v_D01_a_r', 'v_D01_a_i', 'v_D01_b_r', 'v_D01_b_i', 'v_D01_c_r', 'v_D01_c_i', 'v_D01_n_r', 'v_D01_n_i', 'v_D03_a_r', 'v_D03_a_i', 'v_D03_b_r', 'v_D03_b_i', 'v_D03_c_r', 'v_D03_c_i', 'v_D03_n_r', 'v_D03_n_i', 'v_D05_a_r', 'v_D05_a_i', 'v_D05_b_r', 'v_D05_b_i', 'v_D05_c_r', 'v_D05_c_i', 'v_D05_n_r', 'v_D05_n_i', 'v_D08_a_r', 'v_D08_a_i', 'v_D08_b_r', 'v_D08_b_i', 'v_D08_c_r', 'v_D08_c_i', 'v_D08_n_r', 'v_D08_n_i', 'v_D09_a_r', 'v_D09_a_i', 'v_D09_b_r', 'v_D09_b_i', 'v_D09_c_r', 'v_D09_c_i', 'v_D09_n_r', 'v_D09_n_i', 'v_D16_a_r', 'v_D16_a_i', 'v_D16_b_r', 'v_D16_b_i', 'v_D16_c_r', 'v_D16_c_i', 'v_D16_n_r', 'v_D16_n_i', 'v_D19_a_r', 'v_D19_a_i', 'v_D19_b_r', 'v_D19_b_i', 'v_D19_c_r', 'v_D19_c_i', 'v_D19_n_r', 'v_D19_n_i', 'i_l_S01_S03_a_r', 'i_l_S01_S03_a_i', 'i_l_S01_S03_b_r', 'i_l_S01_S03_b_i', 'i_l_S01_S03_c_r', 'i_l_S01_S03_c_i', 'i_l_S01_S03_n_r', 'i_l_S01_S03_n_i', 'i_l_H01_H02_a_r', 'i_l_H01_H02_a_i', 'i_l_H01_H02_b_r', 'i_l_H01_H02_b_i', 'i_l_H01_H02_c_r', 'i_l_H01_H02_c_i', 'i_l_H01_H02_n_r', 'i_l_H01_H02_n_i', 'i_l_D01_D03_a_r', 'i_l_D01_D03_a_i', 'i_l_D01_D03_b_r', 'i_l_D01_D03_b_i', 'i_l_D01_D03_c_r', 'i_l_D01_D03_c_i', 'i_l_D01_D03_n_r', 'i_l_D01_D03_n_i', 'i_load_R01_a_r', 'i_load_R01_a_i', 'i_load_R01_b_r', 'i_load_R01_b_i', 'i_load_R01_c_r', 'i_load_R01_c_i', 'i_load_R01_n_r', 'i_load_R01_n_i', 'i_load_R11_a_r', 'i_load_R11_a_i', 'i_load_R11_b_r', 'i_load_R11_b_i', 'i_load_R11_c_r', 'i_load_R11_c_i', 'i_load_R11_n_r', 'i_load_R11_n_i', 'i_load_R15_a_r', 'i_load_R15_a_i', 'i_load_R15_b_r', 'i_load_R15_b_i', 'i_load_R15_c_r', 'i_load_R15_c_i', 'i_load_R15_n_r', 'i_load_R15_n_i', 'i_load_R16_a_r', 'i_load_R16_a_i', 'i_load_R16_b_r', 'i_load_R16_b_i', 'i_load_R16_c_r', 'i_load_R16_c_i', 'i_load_R16_n_r', 'i_load_R16_n_i', 'i_load_R17_a_r', 'i_load_R17_a_i', 'i_load_R17_b_r', 'i_load_R17_b_i', 'i_load_R17_c_r', 'i_load_R17_c_i', 'i_load_R17_n_r', 'i_load_R17_n_i', 'i_load_R18_a_r', 'i_load_R18_a_i', 'i_load_R18_b_r', 'i_load_R18_b_i', 'i_load_R18_c_r', 'i_load_R18_c_i', 'i_load_R18_n_r', 'i_load_R18_n_i', 'i_load_I02_a_r', 'i_load_I02_a_i', 'i_load_I02_b_r', 'i_load_I02_b_i', 'i_load_I02_c_r', 'i_load_I02_c_i', 'i_load_I02_n_r', 'i_load_I02_n_i', 'i_load_C01_a_r', 'i_load_C01_a_i', 'i_load_C01_b_r', 'i_load_C01_b_i', 'i_load_C01_c_r', 'i_load_C01_c_i', 'i_load_C01_n_r', 'i_load_C01_n_i', 'i_load_C12_a_r', 'i_load_C12_a_i', 'i_load_C12_b_r', 'i_load_C12_b_i', 'i_load_C12_c_r', 'i_load_C12_c_i', 'i_load_C12_n_r', 'i_load_C12_n_i', 'i_load_C13_a_r', 'i_load_C13_a_i', 'i_load_C13_b_r', 'i_load_C13_b_i', 'i_load_C13_c_r', 'i_load_C13_c_i', 'i_load_C13_n_r', 'i_load_C13_n_i', 'i_load_C14_a_r', 'i_load_C14_a_i', 'i_load_C14_b_r', 'i_load_C14_b_i', 'i_load_C14_c_r', 'i_load_C14_c_i', 'i_load_C14_n_r', 'i_load_C14_n_i', 'i_load_C17_a_r', 'i_load_C17_a_i', 'i_load_C17_b_r', 'i_load_C17_b_i', 'i_load_C17_c_r', 'i_load_C17_c_i', 'i_load_C17_n_r', 'i_load_C17_n_i', 'i_load_C18_a_r', 'i_load_C18_a_i', 'i_load_C18_b_r', 'i_load_C18_b_i', 'i_load_C18_c_r', 'i_load_C18_c_i', 'i_load_C18_n_r', 'i_load_C18_n_i', 'i_load_C19_a_r', 'i_load_C19_a_i', 'i_load_C19_b_r', 'i_load_C19_b_i', 'i_load_C19_c_r', 'i_load_C19_c_i', 'i_load_C19_n_r', 'i_load_C19_n_i', 'i_load_C20_a_r', 'i_load_C20_a_i', 'i_load_C20_b_r', 'i_load_C20_b_i', 'i_load_C20_c_r', 'i_load_C20_c_i', 'i_load_C20_n_r', 'i_load_C20_n_i', 'i_load_S15_a_r', 'i_load_S15_a_i', 'i_load_S15_b_r', 'i_load_S15_b_i', 'i_load_S15_c_r', 'i_load_S15_c_i', 'i_load_S15_n_r', 'i_load_S15_n_i', 'i_load_S11_a_r', 'i_load_S11_a_i', 'i_load_S11_b_r', 'i_load_S11_b_i', 'i_load_S11_c_r', 'i_load_S11_c_i', 'i_load_S11_n_r', 'i_load_S11_n_i', 'i_load_S16_a_r', 'i_load_S16_a_i', 'i_load_S16_b_r', 'i_load_S16_b_i', 'i_load_S16_c_r', 'i_load_S16_c_i', 'i_load_S16_n_r', 'i_load_S16_n_i', 'i_load_S17_a_r', 'i_load_S17_a_i', 'i_load_S17_b_r', 'i_load_S17_b_i', 'i_load_S17_c_r', 'i_load_S17_c_i', 'i_load_S17_n_r', 'i_load_S17_n_i', 'i_load_S18_a_r', 'i_load_S18_a_i', 'i_load_S18_b_r', 'i_load_S18_b_i', 'i_load_S18_c_r', 'i_load_S18_c_i', 'i_load_S18_n_r', 'i_load_S18_n_i', 'i_load_H02_a_r', 'i_load_H02_a_i', 'i_load_H02_b_r', 'i_load_H02_b_i', 'i_load_H02_c_r', 'i_load_H02_c_i', 'i_load_H02_n_r', 'i_load_H02_n_i', 'i_load_D11_a_r', 'i_load_D11_a_i', 'i_load_D11_b_r', 'i_load_D11_b_i', 'i_load_D11_c_r', 'i_load_D11_c_i', 'i_load_D11_n_r', 'i_load_D11_n_i', 'i_load_D12_a_r', 'i_load_D12_a_i', 'i_load_D12_b_r', 'i_load_D12_b_i', 'i_load_D12_c_r', 'i_load_D12_c_i', 'i_load_D12_n_r', 'i_load_D12_n_i', 'i_load_D17_a_r', 'i_load_D17_a_i', 'i_load_D17_b_r', 'i_load_D17_b_i', 'i_load_D17_c_r', 'i_load_D17_c_i', 'i_load_D17_n_r', 'i_load_D17_n_i', 'i_load_D20_a_r', 'i_load_D20_a_i', 'i_load_D20_b_r', 'i_load_D20_b_i', 'i_load_D20_c_r', 'i_load_D20_c_i', 'i_load_D20_n_r', 'i_load_D20_n_i', 'i_vsc_R01_a_r', 'i_vsc_R01_a_i', 'i_vsc_R01_b_r', 'i_vsc_R01_b_i', 'i_vsc_R01_c_r', 'i_vsc_R01_c_i', 'i_vsc_R01_n_r', 'i_vsc_R01_n_i', 'i_vsc_S01_a_r', 'i_vsc_S01_n_r', 'p_vsc_S01', 'p_vsc_loss_R01', 'i_vsc_R10_a_r', 'i_vsc_R10_a_i', 'i_vsc_R10_b_r', 'i_vsc_R10_b_i', 'i_vsc_R10_c_r', 'i_vsc_R10_c_i', 'i_vsc_R10_n_r', 'i_vsc_R10_n_i', 'i_vsc_S10_a_r', 'i_vsc_S10_n_r', 'p_vsc_S10', 'p_vsc_loss_R10', 'i_vsc_R14_a_r', 'i_vsc_R14_a_i', 'i_vsc_R14_b_r', 'i_vsc_R14_b_i', 'i_vsc_R14_c_r', 'i_vsc_R14_c_i', 'i_vsc_R14_n_r', 'i_vsc_R14_n_i', 'i_vsc_S14_a_r', 'i_vsc_S14_n_r', 'p_vsc_S14', 'p_vsc_loss_R14', 'p_a_d_I01', 'p_b_d_I01', 'p_c_d_I01', 'p_n_d_I01', 'i_vsc_I01_a_r', 'i_vsc_I01_a_i', 'i_vsc_I01_b_r', 'i_vsc_I01_b_i', 'i_vsc_I01_c_r', 'i_vsc_I01_c_i', 'i_vsc_I01_n_r', 'i_vsc_I01_n_i', 'i_dc_H01', 'p_vsc_H01', 'i_vsc_I02_a_r', 'i_vsc_I02_a_i', 'i_vsc_I02_b_r', 'i_vsc_I02_b_i', 'i_vsc_I02_c_r', 'i_vsc_I02_c_i', 'i_vsc_I02_n_r', 'i_vsc_I02_n_i', 'i_vsc_H02_a_r', 'i_vsc_H02_n_r', 'p_vsc_H02', 'p_vsc_loss_I02', 'i_vsc_C01_a_r', 'i_vsc_C01_a_i', 'i_vsc_C01_b_r', 'i_vsc_C01_b_i', 'i_vsc_C01_c_r', 'i_vsc_C01_c_i', 'i_vsc_C01_n_r', 'i_vsc_C01_n_i', 'i_vsc_D01_a_r', 'i_vsc_D01_n_r', 'p_vsc_D01', 'p_vsc_loss_C01', 'i_vsc_C09_a_r', 'i_vsc_C09_a_i', 'i_vsc_C09_b_r', 'i_vsc_C09_b_i', 'i_vsc_C09_c_r', 'i_vsc_C09_c_i', 'i_vsc_C09_n_r', 'i_vsc_C09_n_i', 'i_vsc_D09_a_r', 'i_vsc_D09_n_r', 'p_vsc_D09', 'p_vsc_loss_C09', 'i_vsc_C11_a_r', 'i_vsc_C11_a_i', 'i_vsc_C11_b_r', 'i_vsc_C11_b_i', 'i_vsc_C11_c_r', 'i_vsc_C11_c_i', 'i_vsc_C11_n_r', 'i_vsc_C11_n_i', 'i_vsc_D11_a_r', 'i_vsc_D11_n_r', 'p_vsc_D11', 'p_vsc_loss_C11', 'i_vsc_C16_a_r', 'i_vsc_C16_a_i', 'i_vsc_C16_b_r', 'i_vsc_C16_b_i', 'i_vsc_C16_c_r', 'i_vsc_C16_c_i', 'i_vsc_C16_n_r', 'i_vsc_C16_n_i', 'i_vsc_D16_a_r', 'i_vsc_D16_n_r', 'p_vsc_D16', 'p_vsc_loss_C16'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['v_R01_a_r', 'v_R01_a_i', 'v_R01_b_r', 'v_R01_b_i', 'v_R01_c_r', 'v_R01_c_i', 'v_R01_n_r', 'v_R01_n_i', 'v_R11_a_r', 'v_R11_a_i', 'v_R11_b_r', 'v_R11_b_i', 'v_R11_c_r', 'v_R11_c_i', 'v_R11_n_r', 'v_R11_n_i', 'v_R15_a_r', 'v_R15_a_i', 'v_R15_b_r', 'v_R15_b_i', 'v_R15_c_r', 'v_R15_c_i', 'v_R15_n_r', 'v_R15_n_i', 'v_R16_a_r', 'v_R16_a_i', 'v_R16_b_r', 'v_R16_b_i', 'v_R16_c_r', 'v_R16_c_i', 'v_R16_n_r', 'v_R16_n_i', 'v_R17_a_r', 'v_R17_a_i', 'v_R17_b_r', 'v_R17_b_i', 'v_R17_c_r', 'v_R17_c_i', 'v_R17_n_r', 'v_R17_n_i', 'v_R18_a_r', 'v_R18_a_i', 'v_R18_b_r', 'v_R18_b_i', 'v_R18_c_r', 'v_R18_c_i', 'v_R18_n_r', 'v_R18_n_i', 'v_I02_a_r', 'v_I02_a_i', 'v_I02_b_r', 'v_I02_b_i', 'v_I02_c_r', 'v_I02_c_i', 'v_I02_n_r', 'v_I02_n_i', 'v_C01_a_r', 'v_C01_a_i', 'v_C01_b_r', 'v_C01_b_i', 'v_C01_c_r', 'v_C01_c_i', 'v_C01_n_r', 'v_C01_n_i', 'v_C12_a_r', 'v_C12_a_i', 'v_C12_b_r', 'v_C12_b_i', 'v_C12_c_r', 'v_C12_c_i', 'v_C12_n_r', 'v_C12_n_i', 'v_C13_a_r', 'v_C13_a_i', 'v_C13_b_r', 'v_C13_b_i', 'v_C13_c_r', 'v_C13_c_i', 'v_C13_n_r', 'v_C13_n_i', 'v_C14_a_r', 'v_C14_a_i', 'v_C14_b_r', 'v_C14_b_i', 'v_C14_c_r', 'v_C14_c_i', 'v_C14_n_r', 'v_C14_n_i', 'v_C17_a_r', 'v_C17_a_i', 'v_C17_b_r', 'v_C17_b_i', 'v_C17_c_r', 'v_C17_c_i', 'v_C17_n_r', 'v_C17_n_i', 'v_C18_a_r', 'v_C18_a_i', 'v_C18_b_r', 'v_C18_b_i', 'v_C18_c_r', 'v_C18_c_i', 'v_C18_n_r', 'v_C18_n_i', 'v_C19_a_r', 'v_C19_a_i', 'v_C19_b_r', 'v_C19_b_i', 'v_C19_c_r', 'v_C19_c_i', 'v_C19_n_r', 'v_C19_n_i', 'v_C20_a_r', 'v_C20_a_i', 'v_C20_b_r', 'v_C20_b_i', 'v_C20_c_r', 'v_C20_c_i', 'v_C20_n_r', 'v_C20_n_i', 'v_S15_a_r', 'v_S15_a_i', 'v_S15_b_r', 'v_S15_b_i', 'v_S15_c_r', 'v_S15_c_i', 'v_S15_n_r', 'v_S15_n_i', 'v_S11_a_r', 'v_S11_a_i', 'v_S11_b_r', 'v_S11_b_i', 'v_S11_c_r', 'v_S11_c_i', 'v_S11_n_r', 'v_S11_n_i', 'v_S16_a_r', 'v_S16_a_i', 'v_S16_b_r', 'v_S16_b_i', 'v_S16_c_r', 'v_S16_c_i', 'v_S16_n_r', 'v_S16_n_i', 'v_S17_a_r', 'v_S17_a_i', 'v_S17_b_r', 'v_S17_b_i', 'v_S17_c_r', 'v_S17_c_i', 'v_S17_n_r', 'v_S17_n_i', 'v_S18_a_r', 'v_S18_a_i', 'v_S18_b_r', 'v_S18_b_i', 'v_S18_c_r', 'v_S18_c_i', 'v_S18_n_r', 'v_S18_n_i', 'v_H02_a_r', 'v_H02_a_i', 'v_H02_b_r', 'v_H02_b_i', 'v_H02_c_r', 'v_H02_c_i', 'v_H02_n_r', 'v_H02_n_i', 'v_D11_a_r', 'v_D11_a_i', 'v_D11_b_r', 'v_D11_b_i', 'v_D11_c_r', 'v_D11_c_i', 'v_D11_n_r', 'v_D11_n_i', 'v_D12_a_r', 'v_D12_a_i', 'v_D12_b_r', 'v_D12_b_i', 'v_D12_c_r', 'v_D12_c_i', 'v_D12_n_r', 'v_D12_n_i', 'v_D17_a_r', 'v_D17_a_i', 'v_D17_b_r', 'v_D17_b_i', 'v_D17_c_r', 'v_D17_c_i', 'v_D17_n_r', 'v_D17_n_i', 'v_D20_a_r', 'v_D20_a_i', 'v_D20_b_r', 'v_D20_b_i', 'v_D20_c_r', 'v_D20_c_i', 'v_D20_n_r', 'v_D20_n_i', 'v_I01_a_r', 'v_I01_a_i', 'v_I01_b_r', 'v_I01_b_i', 'v_I01_c_r', 'v_I01_c_i', 'v_I01_n_r', 'v_I01_n_i', 'v_R02_a_r', 'v_R02_a_i', 'v_R02_b_r', 'v_R02_b_i', 'v_R02_c_r', 'v_R02_c_i', 'v_R02_n_r', 'v_R02_n_i', 'v_R03_a_r', 'v_R03_a_i', 'v_R03_b_r', 'v_R03_b_i', 'v_R03_c_r', 'v_R03_c_i', 'v_R03_n_r', 'v_R03_n_i', 'v_R04_a_r', 'v_R04_a_i', 'v_R04_b_r', 'v_R04_b_i', 'v_R04_c_r', 'v_R04_c_i', 'v_R04_n_r', 'v_R04_n_i', 'v_R05_a_r', 'v_R05_a_i', 'v_R05_b_r', 'v_R05_b_i', 'v_R05_c_r', 'v_R05_c_i', 'v_R05_n_r', 'v_R05_n_i', 'v_R06_a_r', 'v_R06_a_i', 'v_R06_b_r', 'v_R06_b_i', 'v_R06_c_r', 'v_R06_c_i', 'v_R06_n_r', 'v_R06_n_i', 'v_R07_a_r', 'v_R07_a_i', 'v_R07_b_r', 'v_R07_b_i', 'v_R07_c_r', 'v_R07_c_i', 'v_R07_n_r', 'v_R07_n_i', 'v_R08_a_r', 'v_R08_a_i', 'v_R08_b_r', 'v_R08_b_i', 'v_R08_c_r', 'v_R08_c_i', 'v_R08_n_r', 'v_R08_n_i', 'v_R09_a_r', 'v_R09_a_i', 'v_R09_b_r', 'v_R09_b_i', 'v_R09_c_r', 'v_R09_c_i', 'v_R09_n_r', 'v_R09_n_i', 'v_R10_a_r', 'v_R10_a_i', 'v_R10_b_r', 'v_R10_b_i', 'v_R10_c_r', 'v_R10_c_i', 'v_R10_n_r', 'v_R10_n_i', 'v_R12_a_r', 'v_R12_a_i', 'v_R12_b_r', 'v_R12_b_i', 'v_R12_c_r', 'v_R12_c_i', 'v_R12_n_r', 'v_R12_n_i', 'v_R13_a_r', 'v_R13_a_i', 'v_R13_b_r', 'v_R13_b_i', 'v_R13_c_r', 'v_R13_c_i', 'v_R13_n_r', 'v_R13_n_i', 'v_R14_a_r', 'v_R14_a_i', 'v_R14_b_r', 'v_R14_b_i', 'v_R14_c_r', 'v_R14_c_i', 'v_R14_n_r', 'v_R14_n_i', 'v_C02_a_r', 'v_C02_a_i', 'v_C02_b_r', 'v_C02_b_i', 'v_C02_c_r', 'v_C02_c_i', 'v_C02_n_r', 'v_C02_n_i', 'v_C03_a_r', 'v_C03_a_i', 'v_C03_b_r', 'v_C03_b_i', 'v_C03_c_r', 'v_C03_c_i', 'v_C03_n_r', 'v_C03_n_i', 'v_C04_a_r', 'v_C04_a_i', 'v_C04_b_r', 'v_C04_b_i', 'v_C04_c_r', 'v_C04_c_i', 'v_C04_n_r', 'v_C04_n_i', 'v_C05_a_r', 'v_C05_a_i', 'v_C05_b_r', 'v_C05_b_i', 'v_C05_c_r', 'v_C05_c_i', 'v_C05_n_r', 'v_C05_n_i', 'v_C06_a_r', 'v_C06_a_i', 'v_C06_b_r', 'v_C06_b_i', 'v_C06_c_r', 'v_C06_c_i', 'v_C06_n_r', 'v_C06_n_i', 'v_C07_a_r', 'v_C07_a_i', 'v_C07_b_r', 'v_C07_b_i', 'v_C07_c_r', 'v_C07_c_i', 'v_C07_n_r', 'v_C07_n_i', 'v_C08_a_r', 'v_C08_a_i', 'v_C08_b_r', 'v_C08_b_i', 'v_C08_c_r', 'v_C08_c_i', 'v_C08_n_r', 'v_C08_n_i', 'v_C09_a_r', 'v_C09_a_i', 'v_C09_b_r', 'v_C09_b_i', 'v_C09_c_r', 'v_C09_c_i', 'v_C09_n_r', 'v_C09_n_i', 'v_C10_a_r', 'v_C10_a_i', 'v_C10_b_r', 'v_C10_b_i', 'v_C10_c_r', 'v_C10_c_i', 'v_C10_n_r', 'v_C10_n_i', 'v_C11_a_r', 'v_C11_a_i', 'v_C11_b_r', 'v_C11_b_i', 'v_C11_c_r', 'v_C11_c_i', 'v_C11_n_r', 'v_C11_n_i', 'v_C15_a_r', 'v_C15_a_i', 'v_C15_b_r', 'v_C15_b_i', 'v_C15_c_r', 'v_C15_c_i', 'v_C15_n_r', 'v_C15_n_i', 'v_C16_a_r', 'v_C16_a_i', 'v_C16_b_r', 'v_C16_b_i', 'v_C16_c_r', 'v_C16_c_i', 'v_C16_n_r', 'v_C16_n_i', 'v_S01_a_r', 'v_S01_a_i', 'v_S01_b_r', 'v_S01_b_i', 'v_S01_c_r', 'v_S01_c_i', 'v_S01_n_r', 'v_S01_n_i', 'v_S03_a_r', 'v_S03_a_i', 'v_S03_b_r', 'v_S03_b_i', 'v_S03_c_r', 'v_S03_c_i', 'v_S03_n_r', 'v_S03_n_i', 'v_S04_a_r', 'v_S04_a_i', 'v_S04_b_r', 'v_S04_b_i', 'v_S04_c_r', 'v_S04_c_i', 'v_S04_n_r', 'v_S04_n_i', 'v_S06_a_r', 'v_S06_a_i', 'v_S06_b_r', 'v_S06_b_i', 'v_S06_c_r', 'v_S06_c_i', 'v_S06_n_r', 'v_S06_n_i', 'v_S07_a_r', 'v_S07_a_i', 'v_S07_b_r', 'v_S07_b_i', 'v_S07_c_r', 'v_S07_c_i', 'v_S07_n_r', 'v_S07_n_i', 'v_S09_a_r', 'v_S09_a_i', 'v_S09_b_r', 'v_S09_b_i', 'v_S09_c_r', 'v_S09_c_i', 'v_S09_n_r', 'v_S09_n_i', 'v_S10_a_r', 'v_S10_a_i', 'v_S10_b_r', 'v_S10_b_i', 'v_S10_c_r', 'v_S10_c_i', 'v_S10_n_r', 'v_S10_n_i', 'v_S14_a_r', 'v_S14_a_i', 'v_S14_b_r', 'v_S14_b_i', 'v_S14_c_r', 'v_S14_c_i', 'v_S14_n_r', 'v_S14_n_i', 'v_H01_n_r', 'v_H01_n_i', 'v_D01_a_r', 'v_D01_a_i', 'v_D01_b_r', 'v_D01_b_i', 'v_D01_c_r', 'v_D01_c_i', 'v_D01_n_r', 'v_D01_n_i', 'v_D03_a_r', 'v_D03_a_i', 'v_D03_b_r', 'v_D03_b_i', 'v_D03_c_r', 'v_D03_c_i', 'v_D03_n_r', 'v_D03_n_i', 'v_D05_a_r', 'v_D05_a_i', 'v_D05_b_r', 'v_D05_b_i', 'v_D05_c_r', 'v_D05_c_i', 'v_D05_n_r', 'v_D05_n_i', 'v_D08_a_r', 'v_D08_a_i', 'v_D08_b_r', 'v_D08_b_i', 'v_D08_c_r', 'v_D08_c_i', 'v_D08_n_r', 'v_D08_n_i', 'v_D09_a_r', 'v_D09_a_i', 'v_D09_b_r', 'v_D09_b_i', 'v_D09_c_r', 'v_D09_c_i', 'v_D09_n_r', 'v_D09_n_i', 'v_D16_a_r', 'v_D16_a_i', 'v_D16_b_r', 'v_D16_b_i', 'v_D16_c_r', 'v_D16_c_i', 'v_D16_n_r', 'v_D16_n_i', 'v_D19_a_r', 'v_D19_a_i', 'v_D19_b_r', 'v_D19_b_i', 'v_D19_c_r', 'v_D19_c_i', 'v_D19_n_r', 'v_D19_n_i', 'i_l_S01_S03_a_r', 'i_l_S01_S03_a_i', 'i_l_S01_S03_b_r', 'i_l_S01_S03_b_i', 'i_l_S01_S03_c_r', 'i_l_S01_S03_c_i', 'i_l_S01_S03_n_r', 'i_l_S01_S03_n_i', 'i_l_H01_H02_a_r', 'i_l_H01_H02_a_i', 'i_l_H01_H02_b_r', 'i_l_H01_H02_b_i', 'i_l_H01_H02_c_r', 'i_l_H01_H02_c_i', 'i_l_H01_H02_n_r', 'i_l_H01_H02_n_i', 'i_l_D01_D03_a_r', 'i_l_D01_D03_a_i', 'i_l_D01_D03_b_r', 'i_l_D01_D03_b_i', 'i_l_D01_D03_c_r', 'i_l_D01_D03_c_i', 'i_l_D01_D03_n_r', 'i_l_D01_D03_n_i', 'i_load_R01_a_r', 'i_load_R01_a_i', 'i_load_R01_b_r', 'i_load_R01_b_i', 'i_load_R01_c_r', 'i_load_R01_c_i', 'i_load_R01_n_r', 'i_load_R01_n_i', 'i_load_R11_a_r', 'i_load_R11_a_i', 'i_load_R11_b_r', 'i_load_R11_b_i', 'i_load_R11_c_r', 'i_load_R11_c_i', 'i_load_R11_n_r', 'i_load_R11_n_i', 'i_load_R15_a_r', 'i_load_R15_a_i', 'i_load_R15_b_r', 'i_load_R15_b_i', 'i_load_R15_c_r', 'i_load_R15_c_i', 'i_load_R15_n_r', 'i_load_R15_n_i', 'i_load_R16_a_r', 'i_load_R16_a_i', 'i_load_R16_b_r', 'i_load_R16_b_i', 'i_load_R16_c_r', 'i_load_R16_c_i', 'i_load_R16_n_r', 'i_load_R16_n_i', 'i_load_R17_a_r', 'i_load_R17_a_i', 'i_load_R17_b_r', 'i_load_R17_b_i', 'i_load_R17_c_r', 'i_load_R17_c_i', 'i_load_R17_n_r', 'i_load_R17_n_i', 'i_load_R18_a_r', 'i_load_R18_a_i', 'i_load_R18_b_r', 'i_load_R18_b_i', 'i_load_R18_c_r', 'i_load_R18_c_i', 'i_load_R18_n_r', 'i_load_R18_n_i', 'i_load_I02_a_r', 'i_load_I02_a_i', 'i_load_I02_b_r', 'i_load_I02_b_i', 'i_load_I02_c_r', 'i_load_I02_c_i', 'i_load_I02_n_r', 'i_load_I02_n_i', 'i_load_C01_a_r', 'i_load_C01_a_i', 'i_load_C01_b_r', 'i_load_C01_b_i', 'i_load_C01_c_r', 'i_load_C01_c_i', 'i_load_C01_n_r', 'i_load_C01_n_i', 'i_load_C12_a_r', 'i_load_C12_a_i', 'i_load_C12_b_r', 'i_load_C12_b_i', 'i_load_C12_c_r', 'i_load_C12_c_i', 'i_load_C12_n_r', 'i_load_C12_n_i', 'i_load_C13_a_r', 'i_load_C13_a_i', 'i_load_C13_b_r', 'i_load_C13_b_i', 'i_load_C13_c_r', 'i_load_C13_c_i', 'i_load_C13_n_r', 'i_load_C13_n_i', 'i_load_C14_a_r', 'i_load_C14_a_i', 'i_load_C14_b_r', 'i_load_C14_b_i', 'i_load_C14_c_r', 'i_load_C14_c_i', 'i_load_C14_n_r', 'i_load_C14_n_i', 'i_load_C17_a_r', 'i_load_C17_a_i', 'i_load_C17_b_r', 'i_load_C17_b_i', 'i_load_C17_c_r', 'i_load_C17_c_i', 'i_load_C17_n_r', 'i_load_C17_n_i', 'i_load_C18_a_r', 'i_load_C18_a_i', 'i_load_C18_b_r', 'i_load_C18_b_i', 'i_load_C18_c_r', 'i_load_C18_c_i', 'i_load_C18_n_r', 'i_load_C18_n_i', 'i_load_C19_a_r', 'i_load_C19_a_i', 'i_load_C19_b_r', 'i_load_C19_b_i', 'i_load_C19_c_r', 'i_load_C19_c_i', 'i_load_C19_n_r', 'i_load_C19_n_i', 'i_load_C20_a_r', 'i_load_C20_a_i', 'i_load_C20_b_r', 'i_load_C20_b_i', 'i_load_C20_c_r', 'i_load_C20_c_i', 'i_load_C20_n_r', 'i_load_C20_n_i', 'i_load_S15_a_r', 'i_load_S15_a_i', 'i_load_S15_b_r', 'i_load_S15_b_i', 'i_load_S15_c_r', 'i_load_S15_c_i', 'i_load_S15_n_r', 'i_load_S15_n_i', 'i_load_S11_a_r', 'i_load_S11_a_i', 'i_load_S11_b_r', 'i_load_S11_b_i', 'i_load_S11_c_r', 'i_load_S11_c_i', 'i_load_S11_n_r', 'i_load_S11_n_i', 'i_load_S16_a_r', 'i_load_S16_a_i', 'i_load_S16_b_r', 'i_load_S16_b_i', 'i_load_S16_c_r', 'i_load_S16_c_i', 'i_load_S16_n_r', 'i_load_S16_n_i', 'i_load_S17_a_r', 'i_load_S17_a_i', 'i_load_S17_b_r', 'i_load_S17_b_i', 'i_load_S17_c_r', 'i_load_S17_c_i', 'i_load_S17_n_r', 'i_load_S17_n_i', 'i_load_S18_a_r', 'i_load_S18_a_i', 'i_load_S18_b_r', 'i_load_S18_b_i', 'i_load_S18_c_r', 'i_load_S18_c_i', 'i_load_S18_n_r', 'i_load_S18_n_i', 'i_load_H02_a_r', 'i_load_H02_a_i', 'i_load_H02_b_r', 'i_load_H02_b_i', 'i_load_H02_c_r', 'i_load_H02_c_i', 'i_load_H02_n_r', 'i_load_H02_n_i', 'i_load_D11_a_r', 'i_load_D11_a_i', 'i_load_D11_b_r', 'i_load_D11_b_i', 'i_load_D11_c_r', 'i_load_D11_c_i', 'i_load_D11_n_r', 'i_load_D11_n_i', 'i_load_D12_a_r', 'i_load_D12_a_i', 'i_load_D12_b_r', 'i_load_D12_b_i', 'i_load_D12_c_r', 'i_load_D12_c_i', 'i_load_D12_n_r', 'i_load_D12_n_i', 'i_load_D17_a_r', 'i_load_D17_a_i', 'i_load_D17_b_r', 'i_load_D17_b_i', 'i_load_D17_c_r', 'i_load_D17_c_i', 'i_load_D17_n_r', 'i_load_D17_n_i', 'i_load_D20_a_r', 'i_load_D20_a_i', 'i_load_D20_b_r', 'i_load_D20_b_i', 'i_load_D20_c_r', 'i_load_D20_c_i', 'i_load_D20_n_r', 'i_load_D20_n_i', 'i_vsc_R01_a_r', 'i_vsc_R01_a_i', 'i_vsc_R01_b_r', 'i_vsc_R01_b_i', 'i_vsc_R01_c_r', 'i_vsc_R01_c_i', 'i_vsc_R01_n_r', 'i_vsc_R01_n_i', 'i_vsc_S01_a_r', 'i_vsc_S01_n_r', 'p_vsc_S01', 'p_vsc_loss_R01', 'i_vsc_R10_a_r', 'i_vsc_R10_a_i', 'i_vsc_R10_b_r', 'i_vsc_R10_b_i', 'i_vsc_R10_c_r', 'i_vsc_R10_c_i', 'i_vsc_R10_n_r', 'i_vsc_R10_n_i', 'i_vsc_S10_a_r', 'i_vsc_S10_n_r', 'p_vsc_S10', 'p_vsc_loss_R10', 'i_vsc_R14_a_r', 'i_vsc_R14_a_i', 'i_vsc_R14_b_r', 'i_vsc_R14_b_i', 'i_vsc_R14_c_r', 'i_vsc_R14_c_i', 'i_vsc_R14_n_r', 'i_vsc_R14_n_i', 'i_vsc_S14_a_r', 'i_vsc_S14_n_r', 'p_vsc_S14', 'p_vsc_loss_R14', 'p_a_d_I01', 'p_b_d_I01', 'p_c_d_I01', 'p_n_d_I01', 'i_vsc_I01_a_r', 'i_vsc_I01_a_i', 'i_vsc_I01_b_r', 'i_vsc_I01_b_i', 'i_vsc_I01_c_r', 'i_vsc_I01_c_i', 'i_vsc_I01_n_r', 'i_vsc_I01_n_i', 'i_dc_H01', 'p_vsc_H01', 'i_vsc_I02_a_r', 'i_vsc_I02_a_i', 'i_vsc_I02_b_r', 'i_vsc_I02_b_i', 'i_vsc_I02_c_r', 'i_vsc_I02_c_i', 'i_vsc_I02_n_r', 'i_vsc_I02_n_i', 'i_vsc_H02_a_r', 'i_vsc_H02_n_r', 'p_vsc_H02', 'p_vsc_loss_I02', 'i_vsc_C01_a_r', 'i_vsc_C01_a_i', 'i_vsc_C01_b_r', 'i_vsc_C01_b_i', 'i_vsc_C01_c_r', 'i_vsc_C01_c_i', 'i_vsc_C01_n_r', 'i_vsc_C01_n_i', 'i_vsc_D01_a_r', 'i_vsc_D01_n_r', 'p_vsc_D01', 'p_vsc_loss_C01', 'i_vsc_C09_a_r', 'i_vsc_C09_a_i', 'i_vsc_C09_b_r', 'i_vsc_C09_b_i', 'i_vsc_C09_c_r', 'i_vsc_C09_c_i', 'i_vsc_C09_n_r', 'i_vsc_C09_n_i', 'i_vsc_D09_a_r', 'i_vsc_D09_n_r', 'p_vsc_D09', 'p_vsc_loss_C09', 'i_vsc_C11_a_r', 'i_vsc_C11_a_i', 'i_vsc_C11_b_r', 'i_vsc_C11_b_i', 'i_vsc_C11_c_r', 'i_vsc_C11_c_i', 'i_vsc_C11_n_r', 'i_vsc_C11_n_i', 'i_vsc_D11_a_r', 'i_vsc_D11_n_r', 'p_vsc_D11', 'p_vsc_loss_C11', 'i_vsc_C16_a_r', 'i_vsc_C16_a_i', 'i_vsc_C16_b_r', 'i_vsc_C16_b_i', 'i_vsc_C16_c_r', 'i_vsc_C16_c_i', 'i_vsc_C16_n_r', 'i_vsc_C16_n_i', 'i_vsc_D16_a_r', 'i_vsc_D16_n_r', 'p_vsc_D16', 'p_vsc_loss_C16'] 
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
        self.u_ini = np.array(self.inputs_ini_values_list)
        self.p = np.array(self.params_values_list)
        self.xy_0 = np.zeros((self.N_x+self.N_y,))
        self.xy = np.zeros((self.N_x+self.N_y,))
        self.z = np.zeros((self.N_z,))
        
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
        self.sp_jac_ini = sspa.load_npz('./__pycache__/cigre_eu_lv_acdc_opt_sp_jac_ini_num.npz')
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
        #self.sp_jac_run = sspa.csr_matrix((data, self.sp_jac_run_ia, self.sp_jac_run_ja), shape=(self.sp_jac_run_nia,self.sp_jac_run_nja))
        self.sp_jac_run = sspa.load_npz('./__pycache__/cigre_eu_lv_acdc_opt_sp_jac_run_num.npz')
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
        self.sp_jac_trap = sspa.load_npz('./__pycache__/cigre_eu_lv_acdc_opt_sp_jac_trap_num.npz')
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

        self.sp_Fu_run = sspa.load_npz('./__pycache__/cigre_eu_lv_acdc_opt_Fu_run_num.npz')
        self.sp_Gu_run = sspa.load_npz('./__pycache__/cigre_eu_lv_acdc_opt_Gu_run_num.npz')
        self.sp_Hx_run = sspa.load_npz('./__pycache__/cigre_eu_lv_acdc_opt_Hx_run_num.npz')
        self.sp_Hy_run = sspa.load_npz('./__pycache__/cigre_eu_lv_acdc_opt_Hy_run_num.npz')
        self.sp_Hu_run = sspa.load_npz('./__pycache__/cigre_eu_lv_acdc_opt_Hu_run_num.npz')        
 
        



        
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
        
        t,it,it_store,xy = daesolver(t,t_end,it,it_store,xy,u,p,
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
            self.struct[0][item] = self.data[item]
            if item in self.params_list:
                self.params_values_list[self.params_list.index(item)] = self.data[item]
            elif item in self.inputs_ini_list:
                self.inputs_ini_values_list[self.inputs_ini_list.index(item)] = self.data[item]
            elif item in self.inputs_run_list:
                self.inputs_run_values_list[self.inputs_run_list.index(item)] = self.data[item]
            else: 
                print(f'parameter or input {item} not found')

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
        self.iparams_run = np.zeros(10,dtype=np.float64)
    
        t,it,it_store,xy = spdaesolver(t,t_end,it,it_store,xy,u,p,
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
                                  lmax_it=self.lmax_it,ltol=self.ltol,ldamp=self.ldamp,mode=self.mode)
    
        self.t = t
        self.it = it
        self.it_store = it_store
        self.xy = xy
            
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
                 ldamp=self.ldamp)

 
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
def spsstate(xy,u,p,
             J_d,J_i,J_p,
             P_d,P_i,P_p,perm_r,perm_c,
             N_x,N_y,
             max_it=50,tol=1e-8,
             lmax_it=20,ltol=1e-8,ldamp=1.0):
    
   
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
               
        Dxy = sprichardson(J_d,J_i,J_p,-fg,P_d,P_i,P_p,perm_r,perm_c,Dxy,iparams,damp=ldamp,max_it=lmax_it,tol=ltol)
   
        xy += Dxy
        #if np.max(np.abs(fg))<tol: break
        if np.linalg.norm(fg,np.inf)<tol: break

    return xy,it,iparams


    
@numba.njit() 
def daesolver(t,t_end,it,it_store,xy,u,p,jac_trap,T,X,Y,Z,iters,Dt,N_x,N_y,N_z,decimation,max_it=50,itol=1e-8,store=1): 


    fg = np.zeros((N_x+N_y,1),dtype=np.float64)
    fg_i = np.zeros((N_x+N_y),dtype=np.float64)
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    h = np.zeros((N_z),dtype=np.float64)
    
    f_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    h_ptr=ffi.from_buffer(np.ascontiguousarray(h))
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
        h_eval(h_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        it_store = 0  
        T[0] = t 
        X[0,:] = x  
        Y[0,:] = y  
        Z[0,:] = h  

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
                
        h_eval(h_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
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
def spdaesolver(t,t_end,it,it_store,xy,u,p,
                J_d,J_i,J_p,
                P_d,P_i,P_p,perm_r,perm_c,
                T,X,Y,Z,iters,Dt,N_x,N_y,N_z,decimation,
                iparams,
                max_it=50,itol=1e-8,store=1,
                lmax_it=20,ltol=1e-4,ldamp=1.0,mode=0):

    fg_i = np.zeros((N_x+N_y),dtype=np.float64)
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    h = np.zeros((N_z),dtype=np.float64)
    Dxy_i_0 = np.zeros(N_x+N_y,dtype=np.float64) 
    f_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    h_ptr=ffi.from_buffer(np.ascontiguousarray(h))
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
        h_eval(h_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        it_store = 0  
        T[0] = t 
        X[0,:] = x  
        Y[0,:] = y  
        Z[0,:] = h  

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
            Dxy_i = sprichardson(J_d,J_i,J_p,-fg_i,P_d,P_i,P_p,perm_r,perm_c,
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
                
        h_eval(h_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
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

    de_jac_ini_num_eval(jac_ini_ss_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
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

    sp_jac_ini_ia = [0, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 547, 747, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 548, 748, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 549, 749, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 550, 750, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 551, 751, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 552, 752, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 553, 753, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 554, 754, 9, 10, 11, 12, 13, 14, 15, 16, 217, 218, 219, 220, 221, 222, 223, 224, 555, 9, 10, 11, 12, 13, 14, 15, 16, 217, 218, 219, 220, 221, 222, 223, 224, 556, 9, 10, 11, 12, 13, 14, 15, 16, 217, 218, 219, 220, 221, 222, 223, 224, 557, 9, 10, 11, 12, 13, 14, 15, 16, 217, 218, 219, 220, 221, 222, 223, 224, 558, 9, 10, 11, 12, 13, 14, 15, 16, 217, 218, 219, 220, 221, 222, 223, 224, 559, 9, 10, 11, 12, 13, 14, 15, 16, 217, 218, 219, 220, 221, 222, 223, 224, 560, 9, 10, 11, 12, 13, 14, 15, 16, 217, 218, 219, 220, 221, 222, 223, 224, 561, 9, 10, 11, 12, 13, 14, 15, 16, 217, 218, 219, 220, 221, 222, 223, 224, 562, 17, 18, 19, 20, 21, 22, 23, 24, 297, 298, 299, 300, 301, 302, 303, 304, 563, 17, 18, 19, 20, 21, 22, 23, 24, 297, 298, 299, 300, 301, 302, 303, 304, 564, 17, 18, 19, 20, 21, 22, 23, 24, 297, 298, 299, 300, 301, 302, 303, 304, 565, 17, 18, 19, 20, 21, 22, 23, 24, 297, 298, 299, 300, 301, 302, 303, 304, 566, 17, 18, 19, 20, 21, 22, 23, 24, 297, 298, 299, 300, 301, 302, 303, 304, 567, 17, 18, 19, 20, 21, 22, 23, 24, 297, 298, 299, 300, 301, 302, 303, 304, 568, 17, 18, 19, 20, 21, 22, 23, 24, 297, 298, 299, 300, 301, 302, 303, 304, 569, 17, 18, 19, 20, 21, 22, 23, 24, 297, 298, 299, 300, 301, 302, 303, 304, 570, 25, 26, 27, 28, 29, 30, 31, 32, 241, 242, 243, 244, 245, 246, 247, 248, 571, 25, 26, 27, 28, 29, 30, 31, 32, 241, 242, 243, 244, 245, 246, 247, 248, 572, 25, 26, 27, 28, 29, 30, 31, 32, 241, 242, 243, 244, 245, 246, 247, 248, 573, 25, 26, 27, 28, 29, 30, 31, 32, 241, 242, 243, 244, 245, 246, 247, 248, 574, 25, 26, 27, 28, 29, 30, 31, 32, 241, 242, 243, 244, 245, 246, 247, 248, 575, 25, 26, 27, 28, 29, 30, 31, 32, 241, 242, 243, 244, 245, 246, 247, 248, 576, 25, 26, 27, 28, 29, 30, 31, 32, 241, 242, 243, 244, 245, 246, 247, 248, 577, 25, 26, 27, 28, 29, 30, 31, 32, 241, 242, 243, 244, 245, 246, 247, 248, 578, 33, 34, 35, 36, 37, 38, 39, 40, 265, 266, 267, 268, 269, 270, 271, 272, 579, 33, 34, 35, 36, 37, 38, 39, 40, 265, 266, 267, 268, 269, 270, 271, 272, 580, 33, 34, 35, 36, 37, 38, 39, 40, 265, 266, 267, 268, 269, 270, 271, 272, 581, 33, 34, 35, 36, 37, 38, 39, 40, 265, 266, 267, 268, 269, 270, 271, 272, 582, 33, 34, 35, 36, 37, 38, 39, 40, 265, 266, 267, 268, 269, 270, 271, 272, 583, 33, 34, 35, 36, 37, 38, 39, 40, 265, 266, 267, 268, 269, 270, 271, 272, 584, 33, 34, 35, 36, 37, 38, 39, 40, 265, 266, 267, 268, 269, 270, 271, 272, 585, 33, 34, 35, 36, 37, 38, 39, 40, 265, 266, 267, 268, 269, 270, 271, 272, 586, 41, 42, 43, 44, 45, 46, 47, 48, 273, 274, 275, 276, 277, 278, 279, 280, 587, 41, 42, 43, 44, 45, 46, 47, 48, 273, 274, 275, 276, 277, 278, 279, 280, 588, 41, 42, 43, 44, 45, 46, 47, 48, 273, 274, 275, 276, 277, 278, 279, 280, 589, 41, 42, 43, 44, 45, 46, 47, 48, 273, 274, 275, 276, 277, 278, 279, 280, 590, 41, 42, 43, 44, 45, 46, 47, 48, 273, 274, 275, 276, 277, 278, 279, 280, 591, 41, 42, 43, 44, 45, 46, 47, 48, 273, 274, 275, 276, 277, 278, 279, 280, 592, 41, 42, 43, 44, 45, 46, 47, 48, 273, 274, 275, 276, 277, 278, 279, 280, 593, 41, 42, 43, 44, 45, 46, 47, 48, 273, 274, 275, 276, 277, 278, 279, 280, 594, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 595, 797, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 596, 798, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 597, 799, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 598, 800, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 599, 801, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 600, 802, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 601, 803, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 602, 804, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 603, 809, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 604, 810, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 605, 811, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 606, 812, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 607, 813, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 608, 814, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 609, 815, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 610, 816, 65, 66, 67, 68, 69, 70, 71, 72, 377, 378, 379, 380, 381, 382, 383, 384, 611, 65, 66, 67, 68, 69, 70, 71, 72, 377, 378, 379, 380, 381, 382, 383, 384, 612, 65, 66, 67, 68, 69, 70, 71, 72, 377, 378, 379, 380, 381, 382, 383, 384, 613, 65, 66, 67, 68, 69, 70, 71, 72, 377, 378, 379, 380, 381, 382, 383, 384, 614, 65, 66, 67, 68, 69, 70, 71, 72, 377, 378, 379, 380, 381, 382, 383, 384, 615, 65, 66, 67, 68, 69, 70, 71, 72, 377, 378, 379, 380, 381, 382, 383, 384, 616, 65, 66, 67, 68, 69, 70, 71, 72, 377, 378, 379, 380, 381, 382, 383, 384, 617, 65, 66, 67, 68, 69, 70, 71, 72, 377, 378, 379, 380, 381, 382, 383, 384, 618, 73, 74, 75, 76, 77, 78, 79, 80, 377, 378, 379, 380, 381, 382, 383, 384, 619, 73, 74, 75, 76, 77, 78, 79, 80, 377, 378, 379, 380, 381, 382, 383, 384, 620, 73, 74, 75, 76, 77, 78, 79, 80, 377, 378, 379, 380, 381, 382, 383, 384, 621, 73, 74, 75, 76, 77, 78, 79, 80, 377, 378, 379, 380, 381, 382, 383, 384, 622, 73, 74, 75, 76, 77, 78, 79, 80, 377, 378, 379, 380, 381, 382, 383, 384, 623, 73, 74, 75, 76, 77, 78, 79, 80, 377, 378, 379, 380, 381, 382, 383, 384, 624, 73, 74, 75, 76, 77, 78, 79, 80, 377, 378, 379, 380, 381, 382, 383, 384, 625, 73, 74, 75, 76, 77, 78, 79, 80, 377, 378, 379, 380, 381, 382, 383, 384, 626, 81, 82, 83, 84, 85, 86, 87, 88, 369, 370, 371, 372, 373, 374, 375, 376, 627, 81, 82, 83, 84, 85, 86, 87, 88, 369, 370, 371, 372, 373, 374, 375, 376, 628, 81, 82, 83, 84, 85, 86, 87, 88, 369, 370, 371, 372, 373, 374, 375, 376, 629, 81, 82, 83, 84, 85, 86, 87, 88, 369, 370, 371, 372, 373, 374, 375, 376, 630, 81, 82, 83, 84, 85, 86, 87, 88, 369, 370, 371, 372, 373, 374, 375, 376, 631, 81, 82, 83, 84, 85, 86, 87, 88, 369, 370, 371, 372, 373, 374, 375, 376, 632, 81, 82, 83, 84, 85, 86, 87, 88, 369, 370, 371, 372, 373, 374, 375, 376, 633, 81, 82, 83, 84, 85, 86, 87, 88, 369, 370, 371, 372, 373, 374, 375, 376, 634, 89, 90, 91, 92, 93, 94, 95, 96, 393, 394, 395, 396, 397, 398, 399, 400, 635, 89, 90, 91, 92, 93, 94, 95, 96, 393, 394, 395, 396, 397, 398, 399, 400, 636, 89, 90, 91, 92, 93, 94, 95, 96, 393, 394, 395, 396, 397, 398, 399, 400, 637, 89, 90, 91, 92, 93, 94, 95, 96, 393, 394, 395, 396, 397, 398, 399, 400, 638, 89, 90, 91, 92, 93, 94, 95, 96, 393, 394, 395, 396, 397, 398, 399, 400, 639, 89, 90, 91, 92, 93, 94, 95, 96, 393, 394, 395, 396, 397, 398, 399, 400, 640, 89, 90, 91, 92, 93, 94, 95, 96, 393, 394, 395, 396, 397, 398, 399, 400, 641, 89, 90, 91, 92, 93, 94, 95, 96, 393, 394, 395, 396, 397, 398, 399, 400, 642, 97, 98, 99, 100, 101, 102, 103, 104, 385, 386, 387, 388, 389, 390, 391, 392, 643, 97, 98, 99, 100, 101, 102, 103, 104, 385, 386, 387, 388, 389, 390, 391, 392, 644, 97, 98, 99, 100, 101, 102, 103, 104, 385, 386, 387, 388, 389, 390, 391, 392, 645, 97, 98, 99, 100, 101, 102, 103, 104, 385, 386, 387, 388, 389, 390, 391, 392, 646, 97, 98, 99, 100, 101, 102, 103, 104, 385, 386, 387, 388, 389, 390, 391, 392, 647, 97, 98, 99, 100, 101, 102, 103, 104, 385, 386, 387, 388, 389, 390, 391, 392, 648, 97, 98, 99, 100, 101, 102, 103, 104, 385, 386, 387, 388, 389, 390, 391, 392, 649, 97, 98, 99, 100, 101, 102, 103, 104, 385, 386, 387, 388, 389, 390, 391, 392, 650, 105, 106, 107, 108, 109, 110, 111, 112, 353, 354, 355, 356, 357, 358, 359, 360, 651, 105, 106, 107, 108, 109, 110, 111, 112, 353, 354, 355, 356, 357, 358, 359, 360, 652, 105, 106, 107, 108, 109, 110, 111, 112, 353, 354, 355, 356, 357, 358, 359, 360, 653, 105, 106, 107, 108, 109, 110, 111, 112, 353, 354, 355, 356, 357, 358, 359, 360, 654, 105, 106, 107, 108, 109, 110, 111, 112, 353, 354, 355, 356, 357, 358, 359, 360, 655, 105, 106, 107, 108, 109, 110, 111, 112, 353, 354, 355, 356, 357, 358, 359, 360, 656, 105, 106, 107, 108, 109, 110, 111, 112, 353, 354, 355, 356, 357, 358, 359, 360, 657, 105, 106, 107, 108, 109, 110, 111, 112, 353, 354, 355, 356, 357, 358, 359, 360, 658, 113, 114, 115, 116, 117, 118, 119, 120, 361, 362, 363, 364, 365, 366, 367, 368, 659, 113, 114, 115, 116, 117, 118, 119, 120, 361, 362, 363, 364, 365, 366, 367, 368, 660, 113, 114, 115, 116, 117, 118, 119, 120, 361, 362, 363, 364, 365, 366, 367, 368, 661, 113, 114, 115, 116, 117, 118, 119, 120, 361, 362, 363, 364, 365, 366, 367, 368, 662, 113, 114, 115, 116, 117, 118, 119, 120, 361, 362, 363, 364, 365, 366, 367, 368, 663, 113, 114, 115, 116, 117, 118, 119, 120, 361, 362, 363, 364, 365, 366, 367, 368, 664, 113, 114, 115, 116, 117, 118, 119, 120, 361, 362, 363, 364, 365, 366, 367, 368, 665, 113, 114, 115, 116, 117, 118, 119, 120, 361, 362, 363, 364, 365, 366, 367, 368, 666, 121, 457, 667, 122, 458, 668, 123, 459, 669, 124, 460, 670, 125, 461, 671, 126, 462, 672, 127, 463, 673, 128, 464, 674, 129, 409, 675, 130, 410, 676, 131, 411, 677, 132, 412, 678, 133, 413, 679, 134, 414, 680, 135, 415, 681, 136, 416, 682, 137, 425, 683, 138, 426, 684, 139, 427, 685, 140, 428, 686, 141, 429, 687, 142, 430, 688, 143, 431, 689, 144, 432, 690, 145, 441, 691, 146, 442, 692, 147, 443, 693, 148, 444, 694, 149, 445, 695, 150, 446, 696, 151, 447, 697, 152, 448, 698, 153, 449, 699, 154, 450, 700, 155, 451, 701, 156, 452, 702, 157, 453, 703, 158, 454, 704, 159, 455, 705, 160, 456, 706, 161, 433, 515, 707, 805, 162, 434, 516, 708, 163, 435, 517, 709, 164, 436, 518, 710, 165, 437, 519, 711, 166, 438, 520, 712, 167, 439, 465, 521, 713, 806, 168, 440, 466, 522, 714, 169, 177, 475, 715, 841, 170, 178, 476, 716, 171, 179, 477, 717, 172, 180, 478, 718, 173, 181, 479, 719, 174, 182, 480, 720, 175, 183, 481, 721, 842, 176, 184, 482, 722, 169, 177, 723, 170, 178, 724, 171, 179, 725, 172, 180, 726, 173, 181, 727, 174, 182, 728, 175, 183, 729, 176, 184, 730, 185, 507, 731, 186, 508, 732, 187, 509, 733, 188, 510, 734, 189, 511, 735, 190, 512, 736, 191, 513, 737, 192, 514, 738, 193, 499, 739, 194, 500, 740, 195, 501, 741, 196, 502, 742, 197, 503, 743, 198, 504, 744, 199, 505, 745, 200, 506, 746, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 787, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 788, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 789, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 790, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 791, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 792, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 793, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 794, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 9, 10, 11, 12, 13, 14, 15, 16, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 9, 10, 11, 12, 13, 14, 15, 16, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 9, 10, 11, 12, 13, 14, 15, 16, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 9, 10, 11, 12, 13, 14, 15, 16, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 9, 10, 11, 12, 13, 14, 15, 16, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 9, 10, 11, 12, 13, 14, 15, 16, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 9, 10, 11, 12, 13, 14, 15, 16, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 9, 10, 11, 12, 13, 14, 15, 16, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 281, 282, 283, 284, 285, 286, 287, 288, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 281, 282, 283, 284, 285, 286, 287, 288, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 281, 282, 283, 284, 285, 286, 287, 288, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 281, 282, 283, 284, 285, 286, 287, 288, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 281, 282, 283, 284, 285, 286, 287, 288, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 281, 282, 283, 284, 285, 286, 287, 288, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 281, 282, 283, 284, 285, 286, 287, 288, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 281, 282, 283, 284, 285, 286, 287, 288, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 25, 26, 27, 28, 29, 30, 31, 32, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 25, 26, 27, 28, 29, 30, 31, 32, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 25, 26, 27, 28, 29, 30, 31, 32, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 25, 26, 27, 28, 29, 30, 31, 32, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 25, 26, 27, 28, 29, 30, 31, 32, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 25, 26, 27, 28, 29, 30, 31, 32, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 25, 26, 27, 28, 29, 30, 31, 32, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 25, 26, 27, 28, 29, 30, 31, 32, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 33, 34, 35, 36, 37, 38, 39, 40, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 33, 34, 35, 36, 37, 38, 39, 40, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 33, 34, 35, 36, 37, 38, 39, 40, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 33, 34, 35, 36, 37, 38, 39, 40, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 33, 34, 35, 36, 37, 38, 39, 40, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 33, 34, 35, 36, 37, 38, 39, 40, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 33, 34, 35, 36, 37, 38, 39, 40, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 33, 34, 35, 36, 37, 38, 39, 40, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 41, 42, 43, 44, 45, 46, 47, 48, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 759, 41, 42, 43, 44, 45, 46, 47, 48, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 760, 41, 42, 43, 44, 45, 46, 47, 48, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 761, 41, 42, 43, 44, 45, 46, 47, 48, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 762, 41, 42, 43, 44, 45, 46, 47, 48, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 763, 41, 42, 43, 44, 45, 46, 47, 48, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 764, 41, 42, 43, 44, 45, 46, 47, 48, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 765, 41, 42, 43, 44, 45, 46, 47, 48, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 766, 225, 226, 227, 228, 229, 230, 231, 232, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 17, 18, 19, 20, 21, 22, 23, 24, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 771, 17, 18, 19, 20, 21, 22, 23, 24, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 772, 17, 18, 19, 20, 21, 22, 23, 24, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 773, 17, 18, 19, 20, 21, 22, 23, 24, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 774, 17, 18, 19, 20, 21, 22, 23, 24, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 775, 17, 18, 19, 20, 21, 22, 23, 24, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 776, 17, 18, 19, 20, 21, 22, 23, 24, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 777, 17, 18, 19, 20, 21, 22, 23, 24, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 778, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 369, 370, 371, 372, 373, 374, 375, 376, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 369, 370, 371, 372, 373, 374, 375, 376, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 369, 370, 371, 372, 373, 374, 375, 376, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 369, 370, 371, 372, 373, 374, 375, 376, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 369, 370, 371, 372, 373, 374, 375, 376, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 369, 370, 371, 372, 373, 374, 375, 376, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 369, 370, 371, 372, 373, 374, 375, 376, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 369, 370, 371, 372, 373, 374, 375, 376, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 385, 386, 387, 388, 389, 390, 391, 392, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 385, 386, 387, 388, 389, 390, 391, 392, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 385, 386, 387, 388, 389, 390, 391, 392, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 385, 386, 387, 388, 389, 390, 391, 392, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 385, 386, 387, 388, 389, 390, 391, 392, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 385, 386, 387, 388, 389, 390, 391, 392, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 385, 386, 387, 388, 389, 390, 391, 392, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 385, 386, 387, 388, 389, 390, 391, 392, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 105, 106, 107, 108, 109, 110, 111, 112, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 105, 106, 107, 108, 109, 110, 111, 112, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 105, 106, 107, 108, 109, 110, 111, 112, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 105, 106, 107, 108, 109, 110, 111, 112, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 105, 106, 107, 108, 109, 110, 111, 112, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 105, 106, 107, 108, 109, 110, 111, 112, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 105, 106, 107, 108, 109, 110, 111, 112, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 105, 106, 107, 108, 109, 110, 111, 112, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 113, 114, 115, 116, 117, 118, 119, 120, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 821, 113, 114, 115, 116, 117, 118, 119, 120, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 822, 113, 114, 115, 116, 117, 118, 119, 120, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 823, 113, 114, 115, 116, 117, 118, 119, 120, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 824, 113, 114, 115, 116, 117, 118, 119, 120, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 825, 113, 114, 115, 116, 117, 118, 119, 120, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 826, 113, 114, 115, 116, 117, 118, 119, 120, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 827, 113, 114, 115, 116, 117, 118, 119, 120, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 828, 81, 82, 83, 84, 85, 86, 87, 88, 313, 314, 315, 316, 317, 318, 319, 320, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 81, 82, 83, 84, 85, 86, 87, 88, 313, 314, 315, 316, 317, 318, 319, 320, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 81, 82, 83, 84, 85, 86, 87, 88, 313, 314, 315, 316, 317, 318, 319, 320, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 81, 82, 83, 84, 85, 86, 87, 88, 313, 314, 315, 316, 317, 318, 319, 320, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 81, 82, 83, 84, 85, 86, 87, 88, 313, 314, 315, 316, 317, 318, 319, 320, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 81, 82, 83, 84, 85, 86, 87, 88, 313, 314, 315, 316, 317, 318, 319, 320, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 81, 82, 83, 84, 85, 86, 87, 88, 313, 314, 315, 316, 317, 318, 319, 320, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 81, 82, 83, 84, 85, 86, 87, 88, 313, 314, 315, 316, 317, 318, 319, 320, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 833, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 834, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 835, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 836, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 837, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 838, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 839, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 840, 97, 98, 99, 100, 101, 102, 103, 104, 329, 330, 331, 332, 333, 334, 335, 336, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 97, 98, 99, 100, 101, 102, 103, 104, 329, 330, 331, 332, 333, 334, 335, 336, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 97, 98, 99, 100, 101, 102, 103, 104, 329, 330, 331, 332, 333, 334, 335, 336, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 97, 98, 99, 100, 101, 102, 103, 104, 329, 330, 331, 332, 333, 334, 335, 336, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 97, 98, 99, 100, 101, 102, 103, 104, 329, 330, 331, 332, 333, 334, 335, 336, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 97, 98, 99, 100, 101, 102, 103, 104, 329, 330, 331, 332, 333, 334, 335, 336, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 97, 98, 99, 100, 101, 102, 103, 104, 329, 330, 331, 332, 333, 334, 335, 336, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 97, 98, 99, 100, 101, 102, 103, 104, 329, 330, 331, 332, 333, 334, 335, 336, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 89, 90, 91, 92, 93, 94, 95, 96, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 845, 89, 90, 91, 92, 93, 94, 95, 96, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 846, 89, 90, 91, 92, 93, 94, 95, 96, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 847, 89, 90, 91, 92, 93, 94, 95, 96, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 848, 89, 90, 91, 92, 93, 94, 95, 96, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 849, 89, 90, 91, 92, 93, 94, 95, 96, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 850, 89, 90, 91, 92, 93, 94, 95, 96, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 851, 89, 90, 91, 92, 93, 94, 95, 96, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 852, 401, 409, 755, 402, 410, 403, 411, 404, 412, 405, 413, 406, 414, 407, 415, 756, 408, 416, 129, 401, 409, 417, 130, 402, 410, 418, 131, 403, 411, 419, 132, 404, 412, 420, 133, 405, 413, 421, 134, 406, 414, 422, 135, 407, 415, 423, 136, 408, 416, 424, 409, 417, 425, 457, 410, 418, 426, 458, 411, 419, 427, 459, 412, 420, 428, 460, 413, 421, 429, 461, 414, 422, 430, 462, 415, 423, 431, 463, 416, 424, 432, 464, 137, 417, 425, 433, 138, 418, 426, 434, 139, 419, 427, 435, 140, 420, 428, 436, 141, 421, 429, 437, 142, 422, 430, 438, 143, 423, 431, 439, 144, 424, 432, 440, 161, 425, 433, 441, 162, 426, 434, 442, 163, 427, 435, 443, 164, 428, 436, 444, 165, 429, 437, 445, 166, 430, 438, 446, 167, 431, 439, 447, 168, 432, 440, 448, 145, 433, 441, 449, 146, 434, 442, 450, 147, 435, 443, 451, 148, 436, 444, 452, 149, 437, 445, 453, 150, 438, 446, 454, 151, 439, 447, 455, 152, 440, 448, 456, 153, 441, 449, 767, 154, 442, 450, 155, 443, 451, 156, 444, 452, 157, 445, 453, 158, 446, 454, 159, 447, 455, 768, 160, 448, 456, 121, 417, 457, 779, 122, 418, 458, 123, 419, 459, 124, 420, 460, 125, 421, 461, 126, 422, 462, 127, 423, 463, 780, 128, 424, 464, 167, 465, 795, 168, 466, 467, 475, 817, 468, 476, 469, 477, 470, 478, 471, 479, 472, 480, 473, 481, 818, 474, 482, 169, 467, 475, 483, 170, 468, 476, 484, 171, 469, 477, 485, 172, 470, 478, 486, 173, 471, 479, 487, 174, 472, 480, 488, 175, 473, 481, 489, 176, 474, 482, 490, 475, 483, 491, 507, 476, 484, 492, 508, 477, 485, 493, 509, 478, 486, 494, 510, 479, 487, 495, 511, 480, 488, 496, 512, 481, 489, 497, 513, 482, 490, 498, 514, 483, 491, 499, 515, 484, 492, 500, 516, 485, 493, 501, 517, 486, 494, 502, 518, 487, 495, 503, 519, 488, 496, 504, 520, 489, 497, 505, 521, 490, 498, 506, 522, 193, 491, 499, 829, 194, 492, 500, 195, 493, 501, 196, 494, 502, 197, 495, 503, 198, 496, 504, 199, 497, 505, 830, 200, 498, 506, 185, 483, 507, 853, 186, 484, 508, 187, 485, 509, 188, 486, 510, 189, 487, 511, 190, 488, 512, 191, 489, 513, 854, 192, 490, 514, 161, 491, 515, 162, 492, 516, 163, 493, 517, 164, 494, 518, 165, 495, 519, 166, 496, 520, 167, 497, 521, 168, 498, 522, 401, 409, 523, 402, 410, 524, 403, 411, 525, 404, 412, 526, 405, 413, 527, 406, 414, 528, 523, 525, 527, 529, 524, 526, 528, 530, 161, 531, 162, 532, 163, 533, 164, 534, 165, 535, 166, 536, 531, 533, 535, 537, 532, 534, 536, 538, 467, 475, 539, 468, 476, 540, 469, 477, 541, 470, 478, 542, 471, 479, 543, 472, 480, 544, 539, 541, 543, 545, 540, 542, 544, 546, 1, 2, 7, 8, 547, 548, 3, 4, 7, 8, 549, 550, 5, 6, 7, 8, 551, 552, 1, 2, 7, 8, 547, 548, 3, 4, 7, 8, 549, 550, 5, 6, 7, 8, 551, 552, 547, 549, 551, 553, 548, 550, 552, 554, 9, 10, 15, 16, 555, 556, 11, 12, 15, 16, 557, 558, 13, 14, 15, 16, 559, 560, 9, 10, 15, 16, 555, 556, 11, 12, 15, 16, 557, 558, 13, 14, 15, 16, 559, 560, 555, 557, 559, 561, 556, 558, 560, 562, 17, 18, 23, 24, 563, 564, 19, 20, 23, 24, 565, 566, 21, 22, 23, 24, 567, 568, 17, 18, 23, 24, 563, 564, 19, 20, 23, 24, 565, 566, 21, 22, 23, 24, 567, 568, 563, 565, 567, 569, 564, 566, 568, 570, 25, 26, 31, 32, 571, 572, 27, 28, 31, 32, 573, 574, 29, 30, 31, 32, 575, 576, 25, 26, 31, 32, 571, 572, 27, 28, 31, 32, 573, 574, 29, 30, 31, 32, 575, 576, 571, 573, 575, 577, 572, 574, 576, 578, 33, 34, 39, 40, 579, 580, 35, 36, 39, 40, 581, 582, 37, 38, 39, 40, 583, 584, 33, 34, 39, 40, 579, 580, 35, 36, 39, 40, 581, 582, 37, 38, 39, 40, 583, 584, 579, 581, 583, 585, 580, 582, 584, 586, 41, 42, 47, 48, 587, 588, 43, 44, 47, 48, 589, 590, 45, 46, 47, 48, 591, 592, 41, 42, 47, 48, 587, 588, 43, 44, 47, 48, 589, 590, 45, 46, 47, 48, 591, 592, 587, 589, 591, 593, 588, 590, 592, 594, 49, 50, 55, 56, 595, 596, 51, 52, 55, 56, 597, 598, 53, 54, 55, 56, 599, 600, 49, 50, 55, 56, 595, 596, 51, 52, 55, 56, 597, 598, 53, 54, 55, 56, 599, 600, 595, 597, 599, 601, 596, 598, 600, 602, 57, 58, 63, 64, 603, 604, 59, 60, 63, 64, 605, 606, 61, 62, 63, 64, 607, 608, 57, 58, 63, 64, 603, 604, 59, 60, 63, 64, 605, 606, 61, 62, 63, 64, 607, 608, 603, 605, 607, 609, 604, 606, 608, 610, 65, 66, 71, 72, 611, 612, 67, 68, 71, 72, 613, 614, 69, 70, 71, 72, 615, 616, 65, 66, 71, 72, 611, 612, 67, 68, 71, 72, 613, 614, 69, 70, 71, 72, 615, 616, 611, 613, 615, 617, 612, 614, 616, 618, 73, 74, 79, 80, 619, 620, 75, 76, 79, 80, 621, 622, 77, 78, 79, 80, 623, 624, 73, 74, 79, 80, 619, 620, 75, 76, 79, 80, 621, 622, 77, 78, 79, 80, 623, 624, 619, 621, 623, 625, 620, 622, 624, 626, 81, 82, 87, 88, 627, 628, 83, 84, 87, 88, 629, 630, 85, 86, 87, 88, 631, 632, 81, 82, 87, 88, 627, 628, 83, 84, 87, 88, 629, 630, 85, 86, 87, 88, 631, 632, 627, 629, 631, 633, 628, 630, 632, 634, 89, 90, 95, 96, 635, 636, 91, 92, 95, 96, 637, 638, 93, 94, 95, 96, 639, 640, 89, 90, 95, 96, 635, 636, 91, 92, 95, 96, 637, 638, 93, 94, 95, 96, 639, 640, 635, 637, 639, 641, 636, 638, 640, 642, 97, 98, 103, 104, 643, 644, 99, 100, 103, 104, 645, 646, 101, 102, 103, 104, 647, 648, 97, 98, 103, 104, 643, 644, 99, 100, 103, 104, 645, 646, 101, 102, 103, 104, 647, 648, 643, 645, 647, 649, 644, 646, 648, 650, 105, 106, 111, 112, 651, 652, 107, 108, 111, 112, 653, 654, 109, 110, 111, 112, 655, 656, 105, 106, 111, 112, 651, 652, 107, 108, 111, 112, 653, 654, 109, 110, 111, 112, 655, 656, 651, 653, 655, 657, 652, 654, 656, 658, 113, 114, 119, 120, 659, 660, 115, 116, 119, 120, 661, 662, 117, 118, 119, 120, 663, 664, 113, 114, 119, 120, 659, 660, 115, 116, 119, 120, 661, 662, 117, 118, 119, 120, 663, 664, 659, 661, 663, 665, 660, 662, 664, 666, 121, 122, 127, 128, 667, 668, 123, 124, 127, 128, 669, 670, 125, 126, 127, 128, 671, 672, 121, 122, 127, 128, 667, 668, 123, 124, 127, 128, 669, 670, 125, 126, 127, 128, 671, 672, 667, 669, 671, 673, 668, 670, 672, 674, 129, 130, 135, 136, 675, 676, 131, 132, 135, 136, 677, 678, 133, 134, 135, 136, 679, 680, 129, 130, 135, 136, 675, 676, 131, 132, 135, 136, 677, 678, 133, 134, 135, 136, 679, 680, 675, 677, 679, 681, 676, 678, 680, 682, 137, 138, 143, 144, 683, 684, 139, 140, 143, 144, 685, 686, 141, 142, 143, 144, 687, 688, 137, 138, 143, 144, 683, 684, 139, 140, 143, 144, 685, 686, 141, 142, 143, 144, 687, 688, 683, 685, 687, 689, 684, 686, 688, 690, 145, 146, 151, 152, 691, 692, 147, 148, 151, 152, 693, 694, 149, 150, 151, 152, 695, 696, 145, 146, 151, 152, 691, 692, 147, 148, 151, 152, 693, 694, 149, 150, 151, 152, 695, 696, 691, 693, 695, 697, 692, 694, 696, 698, 153, 154, 159, 160, 699, 700, 155, 156, 159, 160, 701, 702, 157, 158, 159, 160, 703, 704, 153, 154, 159, 160, 699, 700, 155, 156, 159, 160, 701, 702, 157, 158, 159, 160, 703, 704, 699, 701, 703, 705, 700, 702, 704, 706, 161, 162, 167, 168, 707, 708, 163, 164, 167, 168, 709, 710, 165, 166, 167, 168, 711, 712, 161, 162, 167, 168, 707, 708, 163, 164, 167, 168, 709, 710, 165, 166, 167, 168, 711, 712, 707, 709, 711, 713, 708, 710, 712, 714, 169, 170, 175, 176, 715, 716, 171, 172, 175, 176, 717, 718, 173, 174, 175, 176, 719, 720, 169, 170, 175, 176, 715, 716, 171, 172, 175, 176, 717, 718, 173, 174, 175, 176, 719, 720, 715, 717, 719, 721, 716, 718, 720, 722, 177, 178, 183, 184, 723, 724, 179, 180, 183, 184, 725, 726, 181, 182, 183, 184, 727, 728, 177, 178, 183, 184, 723, 724, 179, 180, 183, 184, 725, 726, 181, 182, 183, 184, 727, 728, 723, 725, 727, 729, 724, 726, 728, 730, 185, 186, 191, 192, 731, 732, 187, 188, 191, 192, 733, 734, 189, 190, 191, 192, 735, 736, 185, 186, 191, 192, 731, 732, 187, 188, 191, 192, 733, 734, 189, 190, 191, 192, 735, 736, 731, 733, 735, 737, 732, 734, 736, 738, 193, 194, 199, 200, 739, 740, 195, 196, 199, 200, 741, 742, 197, 198, 199, 200, 743, 744, 193, 194, 199, 200, 739, 740, 195, 196, 199, 200, 741, 742, 197, 198, 199, 200, 743, 744, 739, 741, 743, 745, 740, 742, 744, 746, 1, 2, 7, 8, 747, 748, 1, 2, 7, 8, 747, 748, 3, 4, 7, 8, 749, 750, 3, 4, 7, 8, 749, 750, 5, 6, 7, 8, 751, 752, 5, 6, 7, 8, 751, 752, 747, 749, 751, 753, 748, 750, 752, 754, 401, 407, 755, 757, 401, 407, 756, 757, 757, 758, 747, 748, 749, 750, 751, 752, 753, 754, 758, 273, 274, 279, 280, 759, 760, 273, 274, 279, 280, 759, 760, 275, 276, 279, 280, 761, 762, 275, 276, 279, 280, 761, 762, 277, 278, 279, 280, 763, 764, 277, 278, 279, 280, 763, 764, 759, 761, 763, 765, 760, 762, 764, 766, 449, 455, 767, 769, 449, 455, 768, 769, 769, 770, 759, 760, 761, 762, 763, 764, 765, 766, 770, 297, 298, 303, 304, 771, 772, 297, 298, 303, 304, 771, 772, 299, 300, 303, 304, 773, 774, 299, 300, 303, 304, 773, 774, 301, 302, 303, 304, 775, 776, 301, 302, 303, 304, 775, 776, 771, 773, 775, 777, 772, 774, 776, 778, 457, 463, 779, 781, 457, 463, 780, 781, 781, 782, 771, 772, 773, 774, 775, 776, 777, 778, 782, 783, 796, 784, 796, 785, 796, 207, 208, 786, 793, 794, 201, 202, 207, 208, 783, 787, 788, 793, 794, 201, 202, 207, 208, 787, 788, 203, 204, 207, 208, 784, 789, 790, 203, 204, 207, 208, 789, 790, 205, 206, 207, 208, 785, 791, 792, 795, 205, 206, 207, 208, 791, 792, 787, 789, 791, 793, 788, 790, 792, 794, 465, 795, 795, 796, 49, 50, 55, 56, 797, 798, 49, 50, 55, 56, 797, 798, 51, 52, 55, 56, 799, 800, 51, 52, 55, 56, 799, 800, 53, 54, 55, 56, 801, 802, 53, 54, 55, 56, 801, 802, 797, 799, 801, 803, 798, 800, 802, 804, 161, 167, 805, 807, 161, 167, 806, 807, 807, 808, 797, 798, 799, 800, 801, 802, 803, 804, 808, 57, 58, 63, 64, 809, 810, 57, 58, 63, 64, 809, 810, 59, 60, 63, 64, 811, 812, 59, 60, 63, 64, 811, 812, 61, 62, 63, 64, 813, 814, 61, 62, 63, 64, 813, 814, 809, 811, 813, 815, 810, 812, 814, 816, 467, 473, 817, 819, 467, 473, 818, 819, 819, 820, 809, 810, 811, 812, 813, 814, 815, 816, 820, 361, 362, 367, 368, 821, 822, 361, 362, 367, 368, 821, 822, 363, 364, 367, 368, 823, 824, 363, 364, 367, 368, 823, 824, 365, 366, 367, 368, 825, 826, 365, 366, 367, 368, 825, 826, 821, 823, 825, 827, 822, 824, 826, 828, 499, 505, 829, 831, 499, 505, 830, 831, 831, 832, 821, 822, 823, 824, 825, 826, 827, 828, 832, 377, 378, 383, 384, 833, 834, 377, 378, 383, 384, 833, 834, 379, 380, 383, 384, 835, 836, 379, 380, 383, 384, 835, 836, 381, 382, 383, 384, 837, 838, 381, 382, 383, 384, 837, 838, 833, 835, 837, 839, 834, 836, 838, 840, 169, 175, 841, 843, 169, 175, 842, 843, 843, 844, 833, 834, 835, 836, 837, 838, 839, 840, 844, 393, 394, 399, 400, 845, 846, 393, 394, 399, 400, 845, 846, 395, 396, 399, 400, 847, 848, 395, 396, 399, 400, 847, 848, 397, 398, 399, 400, 849, 850, 397, 398, 399, 400, 849, 850, 845, 847, 849, 851, 846, 848, 850, 852, 507, 513, 853, 855, 507, 513, 854, 855, 855, 856, 845, 846, 847, 848, 849, 850, 851, 852, 856]
    sp_jac_ini_ja = [0, 1, 19, 37, 55, 73, 91, 109, 127, 145, 162, 179, 196, 213, 230, 247, 264, 281, 298, 315, 332, 349, 366, 383, 400, 417, 434, 451, 468, 485, 502, 519, 536, 553, 570, 587, 604, 621, 638, 655, 672, 689, 706, 723, 740, 757, 774, 791, 808, 825, 843, 861, 879, 897, 915, 933, 951, 969, 987, 1005, 1023, 1041, 1059, 1077, 1095, 1113, 1130, 1147, 1164, 1181, 1198, 1215, 1232, 1249, 1266, 1283, 1300, 1317, 1334, 1351, 1368, 1385, 1402, 1419, 1436, 1453, 1470, 1487, 1504, 1521, 1538, 1555, 1572, 1589, 1606, 1623, 1640, 1657, 1674, 1691, 1708, 1725, 1742, 1759, 1776, 1793, 1810, 1827, 1844, 1861, 1878, 1895, 1912, 1929, 1946, 1963, 1980, 1997, 2014, 2031, 2048, 2065, 2068, 2071, 2074, 2077, 2080, 2083, 2086, 2089, 2092, 2095, 2098, 2101, 2104, 2107, 2110, 2113, 2116, 2119, 2122, 2125, 2128, 2131, 2134, 2137, 2140, 2143, 2146, 2149, 2152, 2155, 2158, 2161, 2164, 2167, 2170, 2173, 2176, 2179, 2182, 2185, 2190, 2194, 2198, 2202, 2206, 2210, 2216, 2221, 2226, 2230, 2234, 2238, 2242, 2246, 2251, 2255, 2258, 2261, 2264, 2267, 2270, 2273, 2276, 2279, 2282, 2285, 2288, 2291, 2294, 2297, 2300, 2303, 2306, 2309, 2312, 2315, 2318, 2321, 2324, 2327, 2344, 2361, 2378, 2395, 2412, 2429, 2446, 2463, 2487, 2511, 2535, 2559, 2583, 2607, 2631, 2655, 2687, 2719, 2751, 2783, 2815, 2847, 2879, 2911, 2943, 2975, 3007, 3039, 3071, 3103, 3135, 3167, 3191, 3215, 3239, 3263, 3287, 3311, 3335, 3359, 3391, 3423, 3455, 3487, 3519, 3551, 3583, 3615, 3639, 3663, 3687, 3711, 3735, 3759, 3783, 3807, 3831, 3855, 3879, 3903, 3927, 3951, 3975, 3999, 4031, 4063, 4095, 4127, 4159, 4191, 4223, 4255, 4280, 4305, 4330, 4355, 4380, 4405, 4430, 4455, 4479, 4503, 4527, 4551, 4575, 4599, 4623, 4647, 4671, 4695, 4719, 4743, 4767, 4791, 4815, 4839, 4864, 4889, 4914, 4939, 4964, 4989, 5014, 5039, 5063, 5087, 5111, 5135, 5159, 5183, 5207, 5231, 5263, 5295, 5327, 5359, 5391, 5423, 5455, 5487, 5511, 5535, 5559, 5583, 5607, 5631, 5655, 5679, 5711, 5743, 5775, 5807, 5839, 5871, 5903, 5935, 5959, 5983, 6007, 6031, 6055, 6079, 6103, 6127, 6151, 6175, 6199, 6223, 6247, 6271, 6295, 6319, 6351, 6383, 6415, 6447, 6479, 6511, 6543, 6575, 6600, 6625, 6650, 6675, 6700, 6725, 6750, 6775, 6807, 6839, 6871, 6903, 6935, 6967, 6999, 7031, 7064, 7097, 7130, 7163, 7196, 7229, 7262, 7295, 7327, 7359, 7391, 7423, 7455, 7487, 7519, 7551, 7576, 7601, 7626, 7651, 7676, 7701, 7726, 7751, 7754, 7756, 7758, 7760, 7762, 7764, 7767, 7769, 7773, 7777, 7781, 7785, 7789, 7793, 7797, 7801, 7805, 7809, 7813, 7817, 7821, 7825, 7829, 7833, 7837, 7841, 7845, 7849, 7853, 7857, 7861, 7865, 7869, 7873, 7877, 7881, 7885, 7889, 7893, 7897, 7901, 7905, 7909, 7913, 7917, 7921, 7925, 7929, 7933, 7936, 7939, 7942, 7945, 7948, 7952, 7955, 7959, 7962, 7965, 7968, 7971, 7974, 7978, 7981, 7984, 7986, 7989, 7991, 7993, 7995, 7997, 7999, 8002, 8004, 8008, 8012, 8016, 8020, 8024, 8028, 8032, 8036, 8040, 8044, 8048, 8052, 8056, 8060, 8064, 8068, 8072, 8076, 8080, 8084, 8088, 8092, 8096, 8100, 8104, 8107, 8110, 8113, 8116, 8119, 8123, 8126, 8130, 8133, 8136, 8139, 8142, 8145, 8149, 8152, 8155, 8158, 8161, 8164, 8167, 8170, 8173, 8176, 8179, 8182, 8185, 8188, 8191, 8194, 8198, 8202, 8204, 8206, 8208, 8210, 8212, 8214, 8218, 8222, 8225, 8228, 8231, 8234, 8237, 8240, 8244, 8248, 8254, 8260, 8266, 8272, 8278, 8284, 8288, 8292, 8298, 8304, 8310, 8316, 8322, 8328, 8332, 8336, 8342, 8348, 8354, 8360, 8366, 8372, 8376, 8380, 8386, 8392, 8398, 8404, 8410, 8416, 8420, 8424, 8430, 8436, 8442, 8448, 8454, 8460, 8464, 8468, 8474, 8480, 8486, 8492, 8498, 8504, 8508, 8512, 8518, 8524, 8530, 8536, 8542, 8548, 8552, 8556, 8562, 8568, 8574, 8580, 8586, 8592, 8596, 8600, 8606, 8612, 8618, 8624, 8630, 8636, 8640, 8644, 8650, 8656, 8662, 8668, 8674, 8680, 8684, 8688, 8694, 8700, 8706, 8712, 8718, 8724, 8728, 8732, 8738, 8744, 8750, 8756, 8762, 8768, 8772, 8776, 8782, 8788, 8794, 8800, 8806, 8812, 8816, 8820, 8826, 8832, 8838, 8844, 8850, 8856, 8860, 8864, 8870, 8876, 8882, 8888, 8894, 8900, 8904, 8908, 8914, 8920, 8926, 8932, 8938, 8944, 8948, 8952, 8958, 8964, 8970, 8976, 8982, 8988, 8992, 8996, 9002, 9008, 9014, 9020, 9026, 9032, 9036, 9040, 9046, 9052, 9058, 9064, 9070, 9076, 9080, 9084, 9090, 9096, 9102, 9108, 9114, 9120, 9124, 9128, 9134, 9140, 9146, 9152, 9158, 9164, 9168, 9172, 9178, 9184, 9190, 9196, 9202, 9208, 9212, 9216, 9222, 9228, 9234, 9240, 9246, 9252, 9256, 9260, 9266, 9272, 9278, 9284, 9290, 9296, 9300, 9304, 9310, 9316, 9322, 9328, 9334, 9340, 9344, 9348, 9354, 9360, 9366, 9372, 9378, 9384, 9388, 9392, 9396, 9400, 9402, 9411, 9417, 9423, 9429, 9435, 9441, 9447, 9451, 9455, 9459, 9463, 9465, 9474, 9480, 9486, 9492, 9498, 9504, 9510, 9514, 9518, 9522, 9526, 9528, 9537, 9539, 9541, 9543, 9548, 9557, 9563, 9570, 9576, 9584, 9590, 9594, 9598, 9600, 9602, 9608, 9614, 9620, 9626, 9632, 9638, 9642, 9646, 9650, 9654, 9656, 9665, 9671, 9677, 9683, 9689, 9695, 9701, 9705, 9709, 9713, 9717, 9719, 9728, 9734, 9740, 9746, 9752, 9758, 9764, 9768, 9772, 9776, 9780, 9782, 9791, 9797, 9803, 9809, 9815, 9821, 9827, 9831, 9835, 9839, 9843, 9845, 9854, 9860, 9866, 9872, 9878, 9884, 9890, 9894, 9898, 9902, 9906, 9908, 9917]
    sp_jac_ini_nia = 857
    sp_jac_ini_nja = 857
    return sp_jac_ini_ia, sp_jac_ini_ja, sp_jac_ini_nia, sp_jac_ini_nja 

def sp_jac_run_vectors():

    sp_jac_run_ia = [0, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 547, 747, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 548, 748, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 549, 749, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 550, 750, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 551, 751, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 552, 752, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 553, 753, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 554, 754, 9, 10, 11, 12, 13, 14, 15, 16, 217, 218, 219, 220, 221, 222, 223, 224, 555, 9, 10, 11, 12, 13, 14, 15, 16, 217, 218, 219, 220, 221, 222, 223, 224, 556, 9, 10, 11, 12, 13, 14, 15, 16, 217, 218, 219, 220, 221, 222, 223, 224, 557, 9, 10, 11, 12, 13, 14, 15, 16, 217, 218, 219, 220, 221, 222, 223, 224, 558, 9, 10, 11, 12, 13, 14, 15, 16, 217, 218, 219, 220, 221, 222, 223, 224, 559, 9, 10, 11, 12, 13, 14, 15, 16, 217, 218, 219, 220, 221, 222, 223, 224, 560, 9, 10, 11, 12, 13, 14, 15, 16, 217, 218, 219, 220, 221, 222, 223, 224, 561, 9, 10, 11, 12, 13, 14, 15, 16, 217, 218, 219, 220, 221, 222, 223, 224, 562, 17, 18, 19, 20, 21, 22, 23, 24, 297, 298, 299, 300, 301, 302, 303, 304, 563, 17, 18, 19, 20, 21, 22, 23, 24, 297, 298, 299, 300, 301, 302, 303, 304, 564, 17, 18, 19, 20, 21, 22, 23, 24, 297, 298, 299, 300, 301, 302, 303, 304, 565, 17, 18, 19, 20, 21, 22, 23, 24, 297, 298, 299, 300, 301, 302, 303, 304, 566, 17, 18, 19, 20, 21, 22, 23, 24, 297, 298, 299, 300, 301, 302, 303, 304, 567, 17, 18, 19, 20, 21, 22, 23, 24, 297, 298, 299, 300, 301, 302, 303, 304, 568, 17, 18, 19, 20, 21, 22, 23, 24, 297, 298, 299, 300, 301, 302, 303, 304, 569, 17, 18, 19, 20, 21, 22, 23, 24, 297, 298, 299, 300, 301, 302, 303, 304, 570, 25, 26, 27, 28, 29, 30, 31, 32, 241, 242, 243, 244, 245, 246, 247, 248, 571, 25, 26, 27, 28, 29, 30, 31, 32, 241, 242, 243, 244, 245, 246, 247, 248, 572, 25, 26, 27, 28, 29, 30, 31, 32, 241, 242, 243, 244, 245, 246, 247, 248, 573, 25, 26, 27, 28, 29, 30, 31, 32, 241, 242, 243, 244, 245, 246, 247, 248, 574, 25, 26, 27, 28, 29, 30, 31, 32, 241, 242, 243, 244, 245, 246, 247, 248, 575, 25, 26, 27, 28, 29, 30, 31, 32, 241, 242, 243, 244, 245, 246, 247, 248, 576, 25, 26, 27, 28, 29, 30, 31, 32, 241, 242, 243, 244, 245, 246, 247, 248, 577, 25, 26, 27, 28, 29, 30, 31, 32, 241, 242, 243, 244, 245, 246, 247, 248, 578, 33, 34, 35, 36, 37, 38, 39, 40, 265, 266, 267, 268, 269, 270, 271, 272, 579, 33, 34, 35, 36, 37, 38, 39, 40, 265, 266, 267, 268, 269, 270, 271, 272, 580, 33, 34, 35, 36, 37, 38, 39, 40, 265, 266, 267, 268, 269, 270, 271, 272, 581, 33, 34, 35, 36, 37, 38, 39, 40, 265, 266, 267, 268, 269, 270, 271, 272, 582, 33, 34, 35, 36, 37, 38, 39, 40, 265, 266, 267, 268, 269, 270, 271, 272, 583, 33, 34, 35, 36, 37, 38, 39, 40, 265, 266, 267, 268, 269, 270, 271, 272, 584, 33, 34, 35, 36, 37, 38, 39, 40, 265, 266, 267, 268, 269, 270, 271, 272, 585, 33, 34, 35, 36, 37, 38, 39, 40, 265, 266, 267, 268, 269, 270, 271, 272, 586, 41, 42, 43, 44, 45, 46, 47, 48, 273, 274, 275, 276, 277, 278, 279, 280, 587, 41, 42, 43, 44, 45, 46, 47, 48, 273, 274, 275, 276, 277, 278, 279, 280, 588, 41, 42, 43, 44, 45, 46, 47, 48, 273, 274, 275, 276, 277, 278, 279, 280, 589, 41, 42, 43, 44, 45, 46, 47, 48, 273, 274, 275, 276, 277, 278, 279, 280, 590, 41, 42, 43, 44, 45, 46, 47, 48, 273, 274, 275, 276, 277, 278, 279, 280, 591, 41, 42, 43, 44, 45, 46, 47, 48, 273, 274, 275, 276, 277, 278, 279, 280, 592, 41, 42, 43, 44, 45, 46, 47, 48, 273, 274, 275, 276, 277, 278, 279, 280, 593, 41, 42, 43, 44, 45, 46, 47, 48, 273, 274, 275, 276, 277, 278, 279, 280, 594, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 595, 797, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 596, 798, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 597, 799, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 598, 800, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 599, 801, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 600, 802, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 601, 803, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 602, 804, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 603, 809, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 604, 810, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 605, 811, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 606, 812, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 607, 813, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 608, 814, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 609, 815, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 610, 816, 65, 66, 67, 68, 69, 70, 71, 72, 377, 378, 379, 380, 381, 382, 383, 384, 611, 65, 66, 67, 68, 69, 70, 71, 72, 377, 378, 379, 380, 381, 382, 383, 384, 612, 65, 66, 67, 68, 69, 70, 71, 72, 377, 378, 379, 380, 381, 382, 383, 384, 613, 65, 66, 67, 68, 69, 70, 71, 72, 377, 378, 379, 380, 381, 382, 383, 384, 614, 65, 66, 67, 68, 69, 70, 71, 72, 377, 378, 379, 380, 381, 382, 383, 384, 615, 65, 66, 67, 68, 69, 70, 71, 72, 377, 378, 379, 380, 381, 382, 383, 384, 616, 65, 66, 67, 68, 69, 70, 71, 72, 377, 378, 379, 380, 381, 382, 383, 384, 617, 65, 66, 67, 68, 69, 70, 71, 72, 377, 378, 379, 380, 381, 382, 383, 384, 618, 73, 74, 75, 76, 77, 78, 79, 80, 377, 378, 379, 380, 381, 382, 383, 384, 619, 73, 74, 75, 76, 77, 78, 79, 80, 377, 378, 379, 380, 381, 382, 383, 384, 620, 73, 74, 75, 76, 77, 78, 79, 80, 377, 378, 379, 380, 381, 382, 383, 384, 621, 73, 74, 75, 76, 77, 78, 79, 80, 377, 378, 379, 380, 381, 382, 383, 384, 622, 73, 74, 75, 76, 77, 78, 79, 80, 377, 378, 379, 380, 381, 382, 383, 384, 623, 73, 74, 75, 76, 77, 78, 79, 80, 377, 378, 379, 380, 381, 382, 383, 384, 624, 73, 74, 75, 76, 77, 78, 79, 80, 377, 378, 379, 380, 381, 382, 383, 384, 625, 73, 74, 75, 76, 77, 78, 79, 80, 377, 378, 379, 380, 381, 382, 383, 384, 626, 81, 82, 83, 84, 85, 86, 87, 88, 369, 370, 371, 372, 373, 374, 375, 376, 627, 81, 82, 83, 84, 85, 86, 87, 88, 369, 370, 371, 372, 373, 374, 375, 376, 628, 81, 82, 83, 84, 85, 86, 87, 88, 369, 370, 371, 372, 373, 374, 375, 376, 629, 81, 82, 83, 84, 85, 86, 87, 88, 369, 370, 371, 372, 373, 374, 375, 376, 630, 81, 82, 83, 84, 85, 86, 87, 88, 369, 370, 371, 372, 373, 374, 375, 376, 631, 81, 82, 83, 84, 85, 86, 87, 88, 369, 370, 371, 372, 373, 374, 375, 376, 632, 81, 82, 83, 84, 85, 86, 87, 88, 369, 370, 371, 372, 373, 374, 375, 376, 633, 81, 82, 83, 84, 85, 86, 87, 88, 369, 370, 371, 372, 373, 374, 375, 376, 634, 89, 90, 91, 92, 93, 94, 95, 96, 393, 394, 395, 396, 397, 398, 399, 400, 635, 89, 90, 91, 92, 93, 94, 95, 96, 393, 394, 395, 396, 397, 398, 399, 400, 636, 89, 90, 91, 92, 93, 94, 95, 96, 393, 394, 395, 396, 397, 398, 399, 400, 637, 89, 90, 91, 92, 93, 94, 95, 96, 393, 394, 395, 396, 397, 398, 399, 400, 638, 89, 90, 91, 92, 93, 94, 95, 96, 393, 394, 395, 396, 397, 398, 399, 400, 639, 89, 90, 91, 92, 93, 94, 95, 96, 393, 394, 395, 396, 397, 398, 399, 400, 640, 89, 90, 91, 92, 93, 94, 95, 96, 393, 394, 395, 396, 397, 398, 399, 400, 641, 89, 90, 91, 92, 93, 94, 95, 96, 393, 394, 395, 396, 397, 398, 399, 400, 642, 97, 98, 99, 100, 101, 102, 103, 104, 385, 386, 387, 388, 389, 390, 391, 392, 643, 97, 98, 99, 100, 101, 102, 103, 104, 385, 386, 387, 388, 389, 390, 391, 392, 644, 97, 98, 99, 100, 101, 102, 103, 104, 385, 386, 387, 388, 389, 390, 391, 392, 645, 97, 98, 99, 100, 101, 102, 103, 104, 385, 386, 387, 388, 389, 390, 391, 392, 646, 97, 98, 99, 100, 101, 102, 103, 104, 385, 386, 387, 388, 389, 390, 391, 392, 647, 97, 98, 99, 100, 101, 102, 103, 104, 385, 386, 387, 388, 389, 390, 391, 392, 648, 97, 98, 99, 100, 101, 102, 103, 104, 385, 386, 387, 388, 389, 390, 391, 392, 649, 97, 98, 99, 100, 101, 102, 103, 104, 385, 386, 387, 388, 389, 390, 391, 392, 650, 105, 106, 107, 108, 109, 110, 111, 112, 353, 354, 355, 356, 357, 358, 359, 360, 651, 105, 106, 107, 108, 109, 110, 111, 112, 353, 354, 355, 356, 357, 358, 359, 360, 652, 105, 106, 107, 108, 109, 110, 111, 112, 353, 354, 355, 356, 357, 358, 359, 360, 653, 105, 106, 107, 108, 109, 110, 111, 112, 353, 354, 355, 356, 357, 358, 359, 360, 654, 105, 106, 107, 108, 109, 110, 111, 112, 353, 354, 355, 356, 357, 358, 359, 360, 655, 105, 106, 107, 108, 109, 110, 111, 112, 353, 354, 355, 356, 357, 358, 359, 360, 656, 105, 106, 107, 108, 109, 110, 111, 112, 353, 354, 355, 356, 357, 358, 359, 360, 657, 105, 106, 107, 108, 109, 110, 111, 112, 353, 354, 355, 356, 357, 358, 359, 360, 658, 113, 114, 115, 116, 117, 118, 119, 120, 361, 362, 363, 364, 365, 366, 367, 368, 659, 113, 114, 115, 116, 117, 118, 119, 120, 361, 362, 363, 364, 365, 366, 367, 368, 660, 113, 114, 115, 116, 117, 118, 119, 120, 361, 362, 363, 364, 365, 366, 367, 368, 661, 113, 114, 115, 116, 117, 118, 119, 120, 361, 362, 363, 364, 365, 366, 367, 368, 662, 113, 114, 115, 116, 117, 118, 119, 120, 361, 362, 363, 364, 365, 366, 367, 368, 663, 113, 114, 115, 116, 117, 118, 119, 120, 361, 362, 363, 364, 365, 366, 367, 368, 664, 113, 114, 115, 116, 117, 118, 119, 120, 361, 362, 363, 364, 365, 366, 367, 368, 665, 113, 114, 115, 116, 117, 118, 119, 120, 361, 362, 363, 364, 365, 366, 367, 368, 666, 121, 457, 667, 122, 458, 668, 123, 459, 669, 124, 460, 670, 125, 461, 671, 126, 462, 672, 127, 463, 673, 128, 464, 674, 129, 409, 675, 130, 410, 676, 131, 411, 677, 132, 412, 678, 133, 413, 679, 134, 414, 680, 135, 415, 681, 136, 416, 682, 137, 425, 683, 138, 426, 684, 139, 427, 685, 140, 428, 686, 141, 429, 687, 142, 430, 688, 143, 431, 689, 144, 432, 690, 145, 441, 691, 146, 442, 692, 147, 443, 693, 148, 444, 694, 149, 445, 695, 150, 446, 696, 151, 447, 697, 152, 448, 698, 153, 449, 699, 154, 450, 700, 155, 451, 701, 156, 452, 702, 157, 453, 703, 158, 454, 704, 159, 455, 705, 160, 456, 706, 161, 433, 515, 707, 805, 162, 434, 516, 708, 163, 435, 517, 709, 164, 436, 518, 710, 165, 437, 519, 711, 166, 438, 520, 712, 167, 439, 465, 521, 713, 806, 168, 440, 466, 522, 714, 169, 177, 475, 715, 841, 170, 178, 476, 716, 171, 179, 477, 717, 172, 180, 478, 718, 173, 181, 479, 719, 174, 182, 480, 720, 175, 183, 481, 721, 842, 176, 184, 482, 722, 169, 177, 723, 170, 178, 724, 171, 179, 725, 172, 180, 726, 173, 181, 727, 174, 182, 728, 175, 183, 729, 176, 184, 730, 185, 507, 731, 186, 508, 732, 187, 509, 733, 188, 510, 734, 189, 511, 735, 190, 512, 736, 191, 513, 737, 192, 514, 738, 193, 499, 739, 194, 500, 740, 195, 501, 741, 196, 502, 742, 197, 503, 743, 198, 504, 744, 199, 505, 745, 200, 506, 746, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 787, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 788, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 789, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 790, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 791, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 792, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 793, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 794, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 9, 10, 11, 12, 13, 14, 15, 16, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 9, 10, 11, 12, 13, 14, 15, 16, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 9, 10, 11, 12, 13, 14, 15, 16, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 9, 10, 11, 12, 13, 14, 15, 16, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 9, 10, 11, 12, 13, 14, 15, 16, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 9, 10, 11, 12, 13, 14, 15, 16, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 9, 10, 11, 12, 13, 14, 15, 16, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 9, 10, 11, 12, 13, 14, 15, 16, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 281, 282, 283, 284, 285, 286, 287, 288, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 281, 282, 283, 284, 285, 286, 287, 288, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 281, 282, 283, 284, 285, 286, 287, 288, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 281, 282, 283, 284, 285, 286, 287, 288, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 281, 282, 283, 284, 285, 286, 287, 288, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 281, 282, 283, 284, 285, 286, 287, 288, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 281, 282, 283, 284, 285, 286, 287, 288, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 281, 282, 283, 284, 285, 286, 287, 288, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 25, 26, 27, 28, 29, 30, 31, 32, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 25, 26, 27, 28, 29, 30, 31, 32, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 25, 26, 27, 28, 29, 30, 31, 32, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 25, 26, 27, 28, 29, 30, 31, 32, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 25, 26, 27, 28, 29, 30, 31, 32, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 25, 26, 27, 28, 29, 30, 31, 32, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 25, 26, 27, 28, 29, 30, 31, 32, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 25, 26, 27, 28, 29, 30, 31, 32, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 33, 34, 35, 36, 37, 38, 39, 40, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 33, 34, 35, 36, 37, 38, 39, 40, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 33, 34, 35, 36, 37, 38, 39, 40, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 33, 34, 35, 36, 37, 38, 39, 40, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 33, 34, 35, 36, 37, 38, 39, 40, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 33, 34, 35, 36, 37, 38, 39, 40, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 33, 34, 35, 36, 37, 38, 39, 40, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 33, 34, 35, 36, 37, 38, 39, 40, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 41, 42, 43, 44, 45, 46, 47, 48, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 759, 41, 42, 43, 44, 45, 46, 47, 48, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 760, 41, 42, 43, 44, 45, 46, 47, 48, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 761, 41, 42, 43, 44, 45, 46, 47, 48, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 762, 41, 42, 43, 44, 45, 46, 47, 48, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 763, 41, 42, 43, 44, 45, 46, 47, 48, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 764, 41, 42, 43, 44, 45, 46, 47, 48, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 765, 41, 42, 43, 44, 45, 46, 47, 48, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 766, 225, 226, 227, 228, 229, 230, 231, 232, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 17, 18, 19, 20, 21, 22, 23, 24, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 771, 17, 18, 19, 20, 21, 22, 23, 24, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 772, 17, 18, 19, 20, 21, 22, 23, 24, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 773, 17, 18, 19, 20, 21, 22, 23, 24, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 774, 17, 18, 19, 20, 21, 22, 23, 24, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 775, 17, 18, 19, 20, 21, 22, 23, 24, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 776, 17, 18, 19, 20, 21, 22, 23, 24, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 777, 17, 18, 19, 20, 21, 22, 23, 24, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 778, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 369, 370, 371, 372, 373, 374, 375, 376, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 369, 370, 371, 372, 373, 374, 375, 376, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 369, 370, 371, 372, 373, 374, 375, 376, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 369, 370, 371, 372, 373, 374, 375, 376, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 369, 370, 371, 372, 373, 374, 375, 376, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 369, 370, 371, 372, 373, 374, 375, 376, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 369, 370, 371, 372, 373, 374, 375, 376, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 369, 370, 371, 372, 373, 374, 375, 376, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 385, 386, 387, 388, 389, 390, 391, 392, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 385, 386, 387, 388, 389, 390, 391, 392, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 385, 386, 387, 388, 389, 390, 391, 392, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 385, 386, 387, 388, 389, 390, 391, 392, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 385, 386, 387, 388, 389, 390, 391, 392, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 385, 386, 387, 388, 389, 390, 391, 392, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 385, 386, 387, 388, 389, 390, 391, 392, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 385, 386, 387, 388, 389, 390, 391, 392, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 105, 106, 107, 108, 109, 110, 111, 112, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 105, 106, 107, 108, 109, 110, 111, 112, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 105, 106, 107, 108, 109, 110, 111, 112, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 105, 106, 107, 108, 109, 110, 111, 112, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 105, 106, 107, 108, 109, 110, 111, 112, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 105, 106, 107, 108, 109, 110, 111, 112, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 105, 106, 107, 108, 109, 110, 111, 112, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 105, 106, 107, 108, 109, 110, 111, 112, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 113, 114, 115, 116, 117, 118, 119, 120, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 821, 113, 114, 115, 116, 117, 118, 119, 120, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 822, 113, 114, 115, 116, 117, 118, 119, 120, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 823, 113, 114, 115, 116, 117, 118, 119, 120, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 824, 113, 114, 115, 116, 117, 118, 119, 120, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 825, 113, 114, 115, 116, 117, 118, 119, 120, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 826, 113, 114, 115, 116, 117, 118, 119, 120, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 827, 113, 114, 115, 116, 117, 118, 119, 120, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 828, 81, 82, 83, 84, 85, 86, 87, 88, 313, 314, 315, 316, 317, 318, 319, 320, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 81, 82, 83, 84, 85, 86, 87, 88, 313, 314, 315, 316, 317, 318, 319, 320, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 81, 82, 83, 84, 85, 86, 87, 88, 313, 314, 315, 316, 317, 318, 319, 320, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 81, 82, 83, 84, 85, 86, 87, 88, 313, 314, 315, 316, 317, 318, 319, 320, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 81, 82, 83, 84, 85, 86, 87, 88, 313, 314, 315, 316, 317, 318, 319, 320, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 81, 82, 83, 84, 85, 86, 87, 88, 313, 314, 315, 316, 317, 318, 319, 320, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 81, 82, 83, 84, 85, 86, 87, 88, 313, 314, 315, 316, 317, 318, 319, 320, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 81, 82, 83, 84, 85, 86, 87, 88, 313, 314, 315, 316, 317, 318, 319, 320, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 833, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 834, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 835, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 836, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 837, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 838, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 839, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 840, 97, 98, 99, 100, 101, 102, 103, 104, 329, 330, 331, 332, 333, 334, 335, 336, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 97, 98, 99, 100, 101, 102, 103, 104, 329, 330, 331, 332, 333, 334, 335, 336, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 97, 98, 99, 100, 101, 102, 103, 104, 329, 330, 331, 332, 333, 334, 335, 336, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 97, 98, 99, 100, 101, 102, 103, 104, 329, 330, 331, 332, 333, 334, 335, 336, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 97, 98, 99, 100, 101, 102, 103, 104, 329, 330, 331, 332, 333, 334, 335, 336, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 97, 98, 99, 100, 101, 102, 103, 104, 329, 330, 331, 332, 333, 334, 335, 336, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 97, 98, 99, 100, 101, 102, 103, 104, 329, 330, 331, 332, 333, 334, 335, 336, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 97, 98, 99, 100, 101, 102, 103, 104, 329, 330, 331, 332, 333, 334, 335, 336, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 89, 90, 91, 92, 93, 94, 95, 96, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 845, 89, 90, 91, 92, 93, 94, 95, 96, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 846, 89, 90, 91, 92, 93, 94, 95, 96, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 847, 89, 90, 91, 92, 93, 94, 95, 96, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 848, 89, 90, 91, 92, 93, 94, 95, 96, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 849, 89, 90, 91, 92, 93, 94, 95, 96, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 850, 89, 90, 91, 92, 93, 94, 95, 96, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 851, 89, 90, 91, 92, 93, 94, 95, 96, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 852, 401, 409, 755, 402, 410, 403, 411, 404, 412, 405, 413, 406, 414, 407, 415, 756, 408, 416, 129, 401, 409, 417, 130, 402, 410, 418, 131, 403, 411, 419, 132, 404, 412, 420, 133, 405, 413, 421, 134, 406, 414, 422, 135, 407, 415, 423, 136, 408, 416, 424, 409, 417, 425, 457, 410, 418, 426, 458, 411, 419, 427, 459, 412, 420, 428, 460, 413, 421, 429, 461, 414, 422, 430, 462, 415, 423, 431, 463, 416, 424, 432, 464, 137, 417, 425, 433, 138, 418, 426, 434, 139, 419, 427, 435, 140, 420, 428, 436, 141, 421, 429, 437, 142, 422, 430, 438, 143, 423, 431, 439, 144, 424, 432, 440, 161, 425, 433, 441, 162, 426, 434, 442, 163, 427, 435, 443, 164, 428, 436, 444, 165, 429, 437, 445, 166, 430, 438, 446, 167, 431, 439, 447, 168, 432, 440, 448, 145, 433, 441, 449, 146, 434, 442, 450, 147, 435, 443, 451, 148, 436, 444, 452, 149, 437, 445, 453, 150, 438, 446, 454, 151, 439, 447, 455, 152, 440, 448, 456, 153, 441, 449, 767, 154, 442, 450, 155, 443, 451, 156, 444, 452, 157, 445, 453, 158, 446, 454, 159, 447, 455, 768, 160, 448, 456, 121, 417, 457, 779, 122, 418, 458, 123, 419, 459, 124, 420, 460, 125, 421, 461, 126, 422, 462, 127, 423, 463, 780, 128, 424, 464, 167, 465, 795, 168, 466, 467, 475, 817, 468, 476, 469, 477, 470, 478, 471, 479, 472, 480, 473, 481, 818, 474, 482, 169, 467, 475, 483, 170, 468, 476, 484, 171, 469, 477, 485, 172, 470, 478, 486, 173, 471, 479, 487, 174, 472, 480, 488, 175, 473, 481, 489, 176, 474, 482, 490, 475, 483, 491, 507, 476, 484, 492, 508, 477, 485, 493, 509, 478, 486, 494, 510, 479, 487, 495, 511, 480, 488, 496, 512, 481, 489, 497, 513, 482, 490, 498, 514, 483, 491, 499, 515, 484, 492, 500, 516, 485, 493, 501, 517, 486, 494, 502, 518, 487, 495, 503, 519, 488, 496, 504, 520, 489, 497, 505, 521, 490, 498, 506, 522, 193, 491, 499, 829, 194, 492, 500, 195, 493, 501, 196, 494, 502, 197, 495, 503, 198, 496, 504, 199, 497, 505, 830, 200, 498, 506, 185, 483, 507, 853, 186, 484, 508, 187, 485, 509, 188, 486, 510, 189, 487, 511, 190, 488, 512, 191, 489, 513, 854, 192, 490, 514, 161, 491, 515, 162, 492, 516, 163, 493, 517, 164, 494, 518, 165, 495, 519, 166, 496, 520, 167, 497, 521, 168, 498, 522, 401, 409, 523, 402, 410, 524, 403, 411, 525, 404, 412, 526, 405, 413, 527, 406, 414, 528, 523, 525, 527, 529, 524, 526, 528, 530, 161, 531, 162, 532, 163, 533, 164, 534, 165, 535, 166, 536, 531, 533, 535, 537, 532, 534, 536, 538, 467, 475, 539, 468, 476, 540, 469, 477, 541, 470, 478, 542, 471, 479, 543, 472, 480, 544, 539, 541, 543, 545, 540, 542, 544, 546, 1, 2, 7, 8, 547, 548, 3, 4, 7, 8, 549, 550, 5, 6, 7, 8, 551, 552, 1, 2, 7, 8, 547, 548, 3, 4, 7, 8, 549, 550, 5, 6, 7, 8, 551, 552, 547, 549, 551, 553, 548, 550, 552, 554, 9, 10, 15, 16, 555, 556, 11, 12, 15, 16, 557, 558, 13, 14, 15, 16, 559, 560, 9, 10, 15, 16, 555, 556, 11, 12, 15, 16, 557, 558, 13, 14, 15, 16, 559, 560, 555, 557, 559, 561, 556, 558, 560, 562, 17, 18, 23, 24, 563, 564, 19, 20, 23, 24, 565, 566, 21, 22, 23, 24, 567, 568, 17, 18, 23, 24, 563, 564, 19, 20, 23, 24, 565, 566, 21, 22, 23, 24, 567, 568, 563, 565, 567, 569, 564, 566, 568, 570, 25, 26, 31, 32, 571, 572, 27, 28, 31, 32, 573, 574, 29, 30, 31, 32, 575, 576, 25, 26, 31, 32, 571, 572, 27, 28, 31, 32, 573, 574, 29, 30, 31, 32, 575, 576, 571, 573, 575, 577, 572, 574, 576, 578, 33, 34, 39, 40, 579, 580, 35, 36, 39, 40, 581, 582, 37, 38, 39, 40, 583, 584, 33, 34, 39, 40, 579, 580, 35, 36, 39, 40, 581, 582, 37, 38, 39, 40, 583, 584, 579, 581, 583, 585, 580, 582, 584, 586, 41, 42, 47, 48, 587, 588, 43, 44, 47, 48, 589, 590, 45, 46, 47, 48, 591, 592, 41, 42, 47, 48, 587, 588, 43, 44, 47, 48, 589, 590, 45, 46, 47, 48, 591, 592, 587, 589, 591, 593, 588, 590, 592, 594, 49, 50, 55, 56, 595, 596, 51, 52, 55, 56, 597, 598, 53, 54, 55, 56, 599, 600, 49, 50, 55, 56, 595, 596, 51, 52, 55, 56, 597, 598, 53, 54, 55, 56, 599, 600, 595, 597, 599, 601, 596, 598, 600, 602, 57, 58, 63, 64, 603, 604, 59, 60, 63, 64, 605, 606, 61, 62, 63, 64, 607, 608, 57, 58, 63, 64, 603, 604, 59, 60, 63, 64, 605, 606, 61, 62, 63, 64, 607, 608, 603, 605, 607, 609, 604, 606, 608, 610, 65, 66, 71, 72, 611, 612, 67, 68, 71, 72, 613, 614, 69, 70, 71, 72, 615, 616, 65, 66, 71, 72, 611, 612, 67, 68, 71, 72, 613, 614, 69, 70, 71, 72, 615, 616, 611, 613, 615, 617, 612, 614, 616, 618, 73, 74, 79, 80, 619, 620, 75, 76, 79, 80, 621, 622, 77, 78, 79, 80, 623, 624, 73, 74, 79, 80, 619, 620, 75, 76, 79, 80, 621, 622, 77, 78, 79, 80, 623, 624, 619, 621, 623, 625, 620, 622, 624, 626, 81, 82, 87, 88, 627, 628, 83, 84, 87, 88, 629, 630, 85, 86, 87, 88, 631, 632, 81, 82, 87, 88, 627, 628, 83, 84, 87, 88, 629, 630, 85, 86, 87, 88, 631, 632, 627, 629, 631, 633, 628, 630, 632, 634, 89, 90, 95, 96, 635, 636, 91, 92, 95, 96, 637, 638, 93, 94, 95, 96, 639, 640, 89, 90, 95, 96, 635, 636, 91, 92, 95, 96, 637, 638, 93, 94, 95, 96, 639, 640, 635, 637, 639, 641, 636, 638, 640, 642, 97, 98, 103, 104, 643, 644, 99, 100, 103, 104, 645, 646, 101, 102, 103, 104, 647, 648, 97, 98, 103, 104, 643, 644, 99, 100, 103, 104, 645, 646, 101, 102, 103, 104, 647, 648, 643, 645, 647, 649, 644, 646, 648, 650, 105, 106, 111, 112, 651, 652, 107, 108, 111, 112, 653, 654, 109, 110, 111, 112, 655, 656, 105, 106, 111, 112, 651, 652, 107, 108, 111, 112, 653, 654, 109, 110, 111, 112, 655, 656, 651, 653, 655, 657, 652, 654, 656, 658, 113, 114, 119, 120, 659, 660, 115, 116, 119, 120, 661, 662, 117, 118, 119, 120, 663, 664, 113, 114, 119, 120, 659, 660, 115, 116, 119, 120, 661, 662, 117, 118, 119, 120, 663, 664, 659, 661, 663, 665, 660, 662, 664, 666, 121, 122, 127, 128, 667, 668, 123, 124, 127, 128, 669, 670, 125, 126, 127, 128, 671, 672, 121, 122, 127, 128, 667, 668, 123, 124, 127, 128, 669, 670, 125, 126, 127, 128, 671, 672, 667, 669, 671, 673, 668, 670, 672, 674, 129, 130, 135, 136, 675, 676, 131, 132, 135, 136, 677, 678, 133, 134, 135, 136, 679, 680, 129, 130, 135, 136, 675, 676, 131, 132, 135, 136, 677, 678, 133, 134, 135, 136, 679, 680, 675, 677, 679, 681, 676, 678, 680, 682, 137, 138, 143, 144, 683, 684, 139, 140, 143, 144, 685, 686, 141, 142, 143, 144, 687, 688, 137, 138, 143, 144, 683, 684, 139, 140, 143, 144, 685, 686, 141, 142, 143, 144, 687, 688, 683, 685, 687, 689, 684, 686, 688, 690, 145, 146, 151, 152, 691, 692, 147, 148, 151, 152, 693, 694, 149, 150, 151, 152, 695, 696, 145, 146, 151, 152, 691, 692, 147, 148, 151, 152, 693, 694, 149, 150, 151, 152, 695, 696, 691, 693, 695, 697, 692, 694, 696, 698, 153, 154, 159, 160, 699, 700, 155, 156, 159, 160, 701, 702, 157, 158, 159, 160, 703, 704, 153, 154, 159, 160, 699, 700, 155, 156, 159, 160, 701, 702, 157, 158, 159, 160, 703, 704, 699, 701, 703, 705, 700, 702, 704, 706, 161, 162, 167, 168, 707, 708, 163, 164, 167, 168, 709, 710, 165, 166, 167, 168, 711, 712, 161, 162, 167, 168, 707, 708, 163, 164, 167, 168, 709, 710, 165, 166, 167, 168, 711, 712, 707, 709, 711, 713, 708, 710, 712, 714, 169, 170, 175, 176, 715, 716, 171, 172, 175, 176, 717, 718, 173, 174, 175, 176, 719, 720, 169, 170, 175, 176, 715, 716, 171, 172, 175, 176, 717, 718, 173, 174, 175, 176, 719, 720, 715, 717, 719, 721, 716, 718, 720, 722, 177, 178, 183, 184, 723, 724, 179, 180, 183, 184, 725, 726, 181, 182, 183, 184, 727, 728, 177, 178, 183, 184, 723, 724, 179, 180, 183, 184, 725, 726, 181, 182, 183, 184, 727, 728, 723, 725, 727, 729, 724, 726, 728, 730, 185, 186, 191, 192, 731, 732, 187, 188, 191, 192, 733, 734, 189, 190, 191, 192, 735, 736, 185, 186, 191, 192, 731, 732, 187, 188, 191, 192, 733, 734, 189, 190, 191, 192, 735, 736, 731, 733, 735, 737, 732, 734, 736, 738, 193, 194, 199, 200, 739, 740, 195, 196, 199, 200, 741, 742, 197, 198, 199, 200, 743, 744, 193, 194, 199, 200, 739, 740, 195, 196, 199, 200, 741, 742, 197, 198, 199, 200, 743, 744, 739, 741, 743, 745, 740, 742, 744, 746, 1, 2, 7, 8, 747, 748, 1, 2, 7, 8, 747, 748, 3, 4, 7, 8, 749, 750, 3, 4, 7, 8, 749, 750, 5, 6, 7, 8, 751, 752, 5, 6, 7, 8, 751, 752, 747, 749, 751, 753, 748, 750, 752, 754, 401, 407, 755, 757, 401, 407, 756, 757, 757, 758, 747, 748, 749, 750, 751, 752, 753, 754, 758, 273, 274, 279, 280, 759, 760, 273, 274, 279, 280, 759, 760, 275, 276, 279, 280, 761, 762, 275, 276, 279, 280, 761, 762, 277, 278, 279, 280, 763, 764, 277, 278, 279, 280, 763, 764, 759, 761, 763, 765, 760, 762, 764, 766, 449, 455, 767, 769, 449, 455, 768, 769, 769, 770, 759, 760, 761, 762, 763, 764, 765, 766, 770, 297, 298, 303, 304, 771, 772, 297, 298, 303, 304, 771, 772, 299, 300, 303, 304, 773, 774, 299, 300, 303, 304, 773, 774, 301, 302, 303, 304, 775, 776, 301, 302, 303, 304, 775, 776, 771, 773, 775, 777, 772, 774, 776, 778, 457, 463, 779, 781, 457, 463, 780, 781, 781, 782, 771, 772, 773, 774, 775, 776, 777, 778, 782, 783, 796, 784, 796, 785, 796, 207, 208, 786, 793, 794, 201, 202, 207, 208, 783, 787, 788, 793, 794, 201, 202, 207, 208, 787, 788, 203, 204, 207, 208, 784, 789, 790, 203, 204, 207, 208, 789, 790, 205, 206, 207, 208, 785, 791, 792, 795, 205, 206, 207, 208, 791, 792, 787, 789, 791, 793, 788, 790, 792, 794, 465, 795, 795, 796, 49, 50, 55, 56, 797, 798, 49, 50, 55, 56, 797, 798, 51, 52, 55, 56, 799, 800, 51, 52, 55, 56, 799, 800, 53, 54, 55, 56, 801, 802, 53, 54, 55, 56, 801, 802, 797, 799, 801, 803, 798, 800, 802, 804, 161, 167, 805, 807, 161, 167, 806, 807, 807, 808, 797, 798, 799, 800, 801, 802, 803, 804, 808, 57, 58, 63, 64, 809, 810, 57, 58, 63, 64, 809, 810, 59, 60, 63, 64, 811, 812, 59, 60, 63, 64, 811, 812, 61, 62, 63, 64, 813, 814, 61, 62, 63, 64, 813, 814, 809, 811, 813, 815, 810, 812, 814, 816, 467, 473, 817, 819, 467, 473, 818, 819, 819, 820, 809, 810, 811, 812, 813, 814, 815, 816, 820, 361, 362, 367, 368, 821, 822, 361, 362, 367, 368, 821, 822, 363, 364, 367, 368, 823, 824, 363, 364, 367, 368, 823, 824, 365, 366, 367, 368, 825, 826, 365, 366, 367, 368, 825, 826, 821, 823, 825, 827, 822, 824, 826, 828, 499, 505, 829, 831, 499, 505, 830, 831, 831, 832, 821, 822, 823, 824, 825, 826, 827, 828, 832, 377, 378, 383, 384, 833, 834, 377, 378, 383, 384, 833, 834, 379, 380, 383, 384, 835, 836, 379, 380, 383, 384, 835, 836, 381, 382, 383, 384, 837, 838, 381, 382, 383, 384, 837, 838, 833, 835, 837, 839, 834, 836, 838, 840, 169, 175, 841, 843, 169, 175, 842, 843, 843, 844, 833, 834, 835, 836, 837, 838, 839, 840, 844, 393, 394, 399, 400, 845, 846, 393, 394, 399, 400, 845, 846, 395, 396, 399, 400, 847, 848, 395, 396, 399, 400, 847, 848, 397, 398, 399, 400, 849, 850, 397, 398, 399, 400, 849, 850, 845, 847, 849, 851, 846, 848, 850, 852, 507, 513, 853, 855, 507, 513, 854, 855, 855, 856, 845, 846, 847, 848, 849, 850, 851, 852, 856]
    sp_jac_run_ja = [0, 1, 19, 37, 55, 73, 91, 109, 127, 145, 162, 179, 196, 213, 230, 247, 264, 281, 298, 315, 332, 349, 366, 383, 400, 417, 434, 451, 468, 485, 502, 519, 536, 553, 570, 587, 604, 621, 638, 655, 672, 689, 706, 723, 740, 757, 774, 791, 808, 825, 843, 861, 879, 897, 915, 933, 951, 969, 987, 1005, 1023, 1041, 1059, 1077, 1095, 1113, 1130, 1147, 1164, 1181, 1198, 1215, 1232, 1249, 1266, 1283, 1300, 1317, 1334, 1351, 1368, 1385, 1402, 1419, 1436, 1453, 1470, 1487, 1504, 1521, 1538, 1555, 1572, 1589, 1606, 1623, 1640, 1657, 1674, 1691, 1708, 1725, 1742, 1759, 1776, 1793, 1810, 1827, 1844, 1861, 1878, 1895, 1912, 1929, 1946, 1963, 1980, 1997, 2014, 2031, 2048, 2065, 2068, 2071, 2074, 2077, 2080, 2083, 2086, 2089, 2092, 2095, 2098, 2101, 2104, 2107, 2110, 2113, 2116, 2119, 2122, 2125, 2128, 2131, 2134, 2137, 2140, 2143, 2146, 2149, 2152, 2155, 2158, 2161, 2164, 2167, 2170, 2173, 2176, 2179, 2182, 2185, 2190, 2194, 2198, 2202, 2206, 2210, 2216, 2221, 2226, 2230, 2234, 2238, 2242, 2246, 2251, 2255, 2258, 2261, 2264, 2267, 2270, 2273, 2276, 2279, 2282, 2285, 2288, 2291, 2294, 2297, 2300, 2303, 2306, 2309, 2312, 2315, 2318, 2321, 2324, 2327, 2344, 2361, 2378, 2395, 2412, 2429, 2446, 2463, 2487, 2511, 2535, 2559, 2583, 2607, 2631, 2655, 2687, 2719, 2751, 2783, 2815, 2847, 2879, 2911, 2943, 2975, 3007, 3039, 3071, 3103, 3135, 3167, 3191, 3215, 3239, 3263, 3287, 3311, 3335, 3359, 3391, 3423, 3455, 3487, 3519, 3551, 3583, 3615, 3639, 3663, 3687, 3711, 3735, 3759, 3783, 3807, 3831, 3855, 3879, 3903, 3927, 3951, 3975, 3999, 4031, 4063, 4095, 4127, 4159, 4191, 4223, 4255, 4280, 4305, 4330, 4355, 4380, 4405, 4430, 4455, 4479, 4503, 4527, 4551, 4575, 4599, 4623, 4647, 4671, 4695, 4719, 4743, 4767, 4791, 4815, 4839, 4864, 4889, 4914, 4939, 4964, 4989, 5014, 5039, 5063, 5087, 5111, 5135, 5159, 5183, 5207, 5231, 5263, 5295, 5327, 5359, 5391, 5423, 5455, 5487, 5511, 5535, 5559, 5583, 5607, 5631, 5655, 5679, 5711, 5743, 5775, 5807, 5839, 5871, 5903, 5935, 5959, 5983, 6007, 6031, 6055, 6079, 6103, 6127, 6151, 6175, 6199, 6223, 6247, 6271, 6295, 6319, 6351, 6383, 6415, 6447, 6479, 6511, 6543, 6575, 6600, 6625, 6650, 6675, 6700, 6725, 6750, 6775, 6807, 6839, 6871, 6903, 6935, 6967, 6999, 7031, 7064, 7097, 7130, 7163, 7196, 7229, 7262, 7295, 7327, 7359, 7391, 7423, 7455, 7487, 7519, 7551, 7576, 7601, 7626, 7651, 7676, 7701, 7726, 7751, 7754, 7756, 7758, 7760, 7762, 7764, 7767, 7769, 7773, 7777, 7781, 7785, 7789, 7793, 7797, 7801, 7805, 7809, 7813, 7817, 7821, 7825, 7829, 7833, 7837, 7841, 7845, 7849, 7853, 7857, 7861, 7865, 7869, 7873, 7877, 7881, 7885, 7889, 7893, 7897, 7901, 7905, 7909, 7913, 7917, 7921, 7925, 7929, 7933, 7936, 7939, 7942, 7945, 7948, 7952, 7955, 7959, 7962, 7965, 7968, 7971, 7974, 7978, 7981, 7984, 7986, 7989, 7991, 7993, 7995, 7997, 7999, 8002, 8004, 8008, 8012, 8016, 8020, 8024, 8028, 8032, 8036, 8040, 8044, 8048, 8052, 8056, 8060, 8064, 8068, 8072, 8076, 8080, 8084, 8088, 8092, 8096, 8100, 8104, 8107, 8110, 8113, 8116, 8119, 8123, 8126, 8130, 8133, 8136, 8139, 8142, 8145, 8149, 8152, 8155, 8158, 8161, 8164, 8167, 8170, 8173, 8176, 8179, 8182, 8185, 8188, 8191, 8194, 8198, 8202, 8204, 8206, 8208, 8210, 8212, 8214, 8218, 8222, 8225, 8228, 8231, 8234, 8237, 8240, 8244, 8248, 8254, 8260, 8266, 8272, 8278, 8284, 8288, 8292, 8298, 8304, 8310, 8316, 8322, 8328, 8332, 8336, 8342, 8348, 8354, 8360, 8366, 8372, 8376, 8380, 8386, 8392, 8398, 8404, 8410, 8416, 8420, 8424, 8430, 8436, 8442, 8448, 8454, 8460, 8464, 8468, 8474, 8480, 8486, 8492, 8498, 8504, 8508, 8512, 8518, 8524, 8530, 8536, 8542, 8548, 8552, 8556, 8562, 8568, 8574, 8580, 8586, 8592, 8596, 8600, 8606, 8612, 8618, 8624, 8630, 8636, 8640, 8644, 8650, 8656, 8662, 8668, 8674, 8680, 8684, 8688, 8694, 8700, 8706, 8712, 8718, 8724, 8728, 8732, 8738, 8744, 8750, 8756, 8762, 8768, 8772, 8776, 8782, 8788, 8794, 8800, 8806, 8812, 8816, 8820, 8826, 8832, 8838, 8844, 8850, 8856, 8860, 8864, 8870, 8876, 8882, 8888, 8894, 8900, 8904, 8908, 8914, 8920, 8926, 8932, 8938, 8944, 8948, 8952, 8958, 8964, 8970, 8976, 8982, 8988, 8992, 8996, 9002, 9008, 9014, 9020, 9026, 9032, 9036, 9040, 9046, 9052, 9058, 9064, 9070, 9076, 9080, 9084, 9090, 9096, 9102, 9108, 9114, 9120, 9124, 9128, 9134, 9140, 9146, 9152, 9158, 9164, 9168, 9172, 9178, 9184, 9190, 9196, 9202, 9208, 9212, 9216, 9222, 9228, 9234, 9240, 9246, 9252, 9256, 9260, 9266, 9272, 9278, 9284, 9290, 9296, 9300, 9304, 9310, 9316, 9322, 9328, 9334, 9340, 9344, 9348, 9354, 9360, 9366, 9372, 9378, 9384, 9388, 9392, 9396, 9400, 9402, 9411, 9417, 9423, 9429, 9435, 9441, 9447, 9451, 9455, 9459, 9463, 9465, 9474, 9480, 9486, 9492, 9498, 9504, 9510, 9514, 9518, 9522, 9526, 9528, 9537, 9539, 9541, 9543, 9548, 9557, 9563, 9570, 9576, 9584, 9590, 9594, 9598, 9600, 9602, 9608, 9614, 9620, 9626, 9632, 9638, 9642, 9646, 9650, 9654, 9656, 9665, 9671, 9677, 9683, 9689, 9695, 9701, 9705, 9709, 9713, 9717, 9719, 9728, 9734, 9740, 9746, 9752, 9758, 9764, 9768, 9772, 9776, 9780, 9782, 9791, 9797, 9803, 9809, 9815, 9821, 9827, 9831, 9835, 9839, 9843, 9845, 9854, 9860, 9866, 9872, 9878, 9884, 9890, 9894, 9898, 9902, 9906, 9908, 9917]
    sp_jac_run_nia = 857
    sp_jac_run_nja = 857
    return sp_jac_run_ia, sp_jac_run_ja, sp_jac_run_nia, sp_jac_run_nja 

def sp_jac_trap_vectors():

    sp_jac_trap_ia = [0, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 547, 747, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 548, 748, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 549, 749, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 550, 750, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 551, 751, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 552, 752, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 553, 753, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 554, 754, 9, 10, 11, 12, 13, 14, 15, 16, 217, 218, 219, 220, 221, 222, 223, 224, 555, 9, 10, 11, 12, 13, 14, 15, 16, 217, 218, 219, 220, 221, 222, 223, 224, 556, 9, 10, 11, 12, 13, 14, 15, 16, 217, 218, 219, 220, 221, 222, 223, 224, 557, 9, 10, 11, 12, 13, 14, 15, 16, 217, 218, 219, 220, 221, 222, 223, 224, 558, 9, 10, 11, 12, 13, 14, 15, 16, 217, 218, 219, 220, 221, 222, 223, 224, 559, 9, 10, 11, 12, 13, 14, 15, 16, 217, 218, 219, 220, 221, 222, 223, 224, 560, 9, 10, 11, 12, 13, 14, 15, 16, 217, 218, 219, 220, 221, 222, 223, 224, 561, 9, 10, 11, 12, 13, 14, 15, 16, 217, 218, 219, 220, 221, 222, 223, 224, 562, 17, 18, 19, 20, 21, 22, 23, 24, 297, 298, 299, 300, 301, 302, 303, 304, 563, 17, 18, 19, 20, 21, 22, 23, 24, 297, 298, 299, 300, 301, 302, 303, 304, 564, 17, 18, 19, 20, 21, 22, 23, 24, 297, 298, 299, 300, 301, 302, 303, 304, 565, 17, 18, 19, 20, 21, 22, 23, 24, 297, 298, 299, 300, 301, 302, 303, 304, 566, 17, 18, 19, 20, 21, 22, 23, 24, 297, 298, 299, 300, 301, 302, 303, 304, 567, 17, 18, 19, 20, 21, 22, 23, 24, 297, 298, 299, 300, 301, 302, 303, 304, 568, 17, 18, 19, 20, 21, 22, 23, 24, 297, 298, 299, 300, 301, 302, 303, 304, 569, 17, 18, 19, 20, 21, 22, 23, 24, 297, 298, 299, 300, 301, 302, 303, 304, 570, 25, 26, 27, 28, 29, 30, 31, 32, 241, 242, 243, 244, 245, 246, 247, 248, 571, 25, 26, 27, 28, 29, 30, 31, 32, 241, 242, 243, 244, 245, 246, 247, 248, 572, 25, 26, 27, 28, 29, 30, 31, 32, 241, 242, 243, 244, 245, 246, 247, 248, 573, 25, 26, 27, 28, 29, 30, 31, 32, 241, 242, 243, 244, 245, 246, 247, 248, 574, 25, 26, 27, 28, 29, 30, 31, 32, 241, 242, 243, 244, 245, 246, 247, 248, 575, 25, 26, 27, 28, 29, 30, 31, 32, 241, 242, 243, 244, 245, 246, 247, 248, 576, 25, 26, 27, 28, 29, 30, 31, 32, 241, 242, 243, 244, 245, 246, 247, 248, 577, 25, 26, 27, 28, 29, 30, 31, 32, 241, 242, 243, 244, 245, 246, 247, 248, 578, 33, 34, 35, 36, 37, 38, 39, 40, 265, 266, 267, 268, 269, 270, 271, 272, 579, 33, 34, 35, 36, 37, 38, 39, 40, 265, 266, 267, 268, 269, 270, 271, 272, 580, 33, 34, 35, 36, 37, 38, 39, 40, 265, 266, 267, 268, 269, 270, 271, 272, 581, 33, 34, 35, 36, 37, 38, 39, 40, 265, 266, 267, 268, 269, 270, 271, 272, 582, 33, 34, 35, 36, 37, 38, 39, 40, 265, 266, 267, 268, 269, 270, 271, 272, 583, 33, 34, 35, 36, 37, 38, 39, 40, 265, 266, 267, 268, 269, 270, 271, 272, 584, 33, 34, 35, 36, 37, 38, 39, 40, 265, 266, 267, 268, 269, 270, 271, 272, 585, 33, 34, 35, 36, 37, 38, 39, 40, 265, 266, 267, 268, 269, 270, 271, 272, 586, 41, 42, 43, 44, 45, 46, 47, 48, 273, 274, 275, 276, 277, 278, 279, 280, 587, 41, 42, 43, 44, 45, 46, 47, 48, 273, 274, 275, 276, 277, 278, 279, 280, 588, 41, 42, 43, 44, 45, 46, 47, 48, 273, 274, 275, 276, 277, 278, 279, 280, 589, 41, 42, 43, 44, 45, 46, 47, 48, 273, 274, 275, 276, 277, 278, 279, 280, 590, 41, 42, 43, 44, 45, 46, 47, 48, 273, 274, 275, 276, 277, 278, 279, 280, 591, 41, 42, 43, 44, 45, 46, 47, 48, 273, 274, 275, 276, 277, 278, 279, 280, 592, 41, 42, 43, 44, 45, 46, 47, 48, 273, 274, 275, 276, 277, 278, 279, 280, 593, 41, 42, 43, 44, 45, 46, 47, 48, 273, 274, 275, 276, 277, 278, 279, 280, 594, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 595, 797, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 596, 798, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 597, 799, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 598, 800, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 599, 801, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 600, 802, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 601, 803, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 602, 804, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 603, 809, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 604, 810, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 605, 811, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 606, 812, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 607, 813, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 608, 814, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 609, 815, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 610, 816, 65, 66, 67, 68, 69, 70, 71, 72, 377, 378, 379, 380, 381, 382, 383, 384, 611, 65, 66, 67, 68, 69, 70, 71, 72, 377, 378, 379, 380, 381, 382, 383, 384, 612, 65, 66, 67, 68, 69, 70, 71, 72, 377, 378, 379, 380, 381, 382, 383, 384, 613, 65, 66, 67, 68, 69, 70, 71, 72, 377, 378, 379, 380, 381, 382, 383, 384, 614, 65, 66, 67, 68, 69, 70, 71, 72, 377, 378, 379, 380, 381, 382, 383, 384, 615, 65, 66, 67, 68, 69, 70, 71, 72, 377, 378, 379, 380, 381, 382, 383, 384, 616, 65, 66, 67, 68, 69, 70, 71, 72, 377, 378, 379, 380, 381, 382, 383, 384, 617, 65, 66, 67, 68, 69, 70, 71, 72, 377, 378, 379, 380, 381, 382, 383, 384, 618, 73, 74, 75, 76, 77, 78, 79, 80, 377, 378, 379, 380, 381, 382, 383, 384, 619, 73, 74, 75, 76, 77, 78, 79, 80, 377, 378, 379, 380, 381, 382, 383, 384, 620, 73, 74, 75, 76, 77, 78, 79, 80, 377, 378, 379, 380, 381, 382, 383, 384, 621, 73, 74, 75, 76, 77, 78, 79, 80, 377, 378, 379, 380, 381, 382, 383, 384, 622, 73, 74, 75, 76, 77, 78, 79, 80, 377, 378, 379, 380, 381, 382, 383, 384, 623, 73, 74, 75, 76, 77, 78, 79, 80, 377, 378, 379, 380, 381, 382, 383, 384, 624, 73, 74, 75, 76, 77, 78, 79, 80, 377, 378, 379, 380, 381, 382, 383, 384, 625, 73, 74, 75, 76, 77, 78, 79, 80, 377, 378, 379, 380, 381, 382, 383, 384, 626, 81, 82, 83, 84, 85, 86, 87, 88, 369, 370, 371, 372, 373, 374, 375, 376, 627, 81, 82, 83, 84, 85, 86, 87, 88, 369, 370, 371, 372, 373, 374, 375, 376, 628, 81, 82, 83, 84, 85, 86, 87, 88, 369, 370, 371, 372, 373, 374, 375, 376, 629, 81, 82, 83, 84, 85, 86, 87, 88, 369, 370, 371, 372, 373, 374, 375, 376, 630, 81, 82, 83, 84, 85, 86, 87, 88, 369, 370, 371, 372, 373, 374, 375, 376, 631, 81, 82, 83, 84, 85, 86, 87, 88, 369, 370, 371, 372, 373, 374, 375, 376, 632, 81, 82, 83, 84, 85, 86, 87, 88, 369, 370, 371, 372, 373, 374, 375, 376, 633, 81, 82, 83, 84, 85, 86, 87, 88, 369, 370, 371, 372, 373, 374, 375, 376, 634, 89, 90, 91, 92, 93, 94, 95, 96, 393, 394, 395, 396, 397, 398, 399, 400, 635, 89, 90, 91, 92, 93, 94, 95, 96, 393, 394, 395, 396, 397, 398, 399, 400, 636, 89, 90, 91, 92, 93, 94, 95, 96, 393, 394, 395, 396, 397, 398, 399, 400, 637, 89, 90, 91, 92, 93, 94, 95, 96, 393, 394, 395, 396, 397, 398, 399, 400, 638, 89, 90, 91, 92, 93, 94, 95, 96, 393, 394, 395, 396, 397, 398, 399, 400, 639, 89, 90, 91, 92, 93, 94, 95, 96, 393, 394, 395, 396, 397, 398, 399, 400, 640, 89, 90, 91, 92, 93, 94, 95, 96, 393, 394, 395, 396, 397, 398, 399, 400, 641, 89, 90, 91, 92, 93, 94, 95, 96, 393, 394, 395, 396, 397, 398, 399, 400, 642, 97, 98, 99, 100, 101, 102, 103, 104, 385, 386, 387, 388, 389, 390, 391, 392, 643, 97, 98, 99, 100, 101, 102, 103, 104, 385, 386, 387, 388, 389, 390, 391, 392, 644, 97, 98, 99, 100, 101, 102, 103, 104, 385, 386, 387, 388, 389, 390, 391, 392, 645, 97, 98, 99, 100, 101, 102, 103, 104, 385, 386, 387, 388, 389, 390, 391, 392, 646, 97, 98, 99, 100, 101, 102, 103, 104, 385, 386, 387, 388, 389, 390, 391, 392, 647, 97, 98, 99, 100, 101, 102, 103, 104, 385, 386, 387, 388, 389, 390, 391, 392, 648, 97, 98, 99, 100, 101, 102, 103, 104, 385, 386, 387, 388, 389, 390, 391, 392, 649, 97, 98, 99, 100, 101, 102, 103, 104, 385, 386, 387, 388, 389, 390, 391, 392, 650, 105, 106, 107, 108, 109, 110, 111, 112, 353, 354, 355, 356, 357, 358, 359, 360, 651, 105, 106, 107, 108, 109, 110, 111, 112, 353, 354, 355, 356, 357, 358, 359, 360, 652, 105, 106, 107, 108, 109, 110, 111, 112, 353, 354, 355, 356, 357, 358, 359, 360, 653, 105, 106, 107, 108, 109, 110, 111, 112, 353, 354, 355, 356, 357, 358, 359, 360, 654, 105, 106, 107, 108, 109, 110, 111, 112, 353, 354, 355, 356, 357, 358, 359, 360, 655, 105, 106, 107, 108, 109, 110, 111, 112, 353, 354, 355, 356, 357, 358, 359, 360, 656, 105, 106, 107, 108, 109, 110, 111, 112, 353, 354, 355, 356, 357, 358, 359, 360, 657, 105, 106, 107, 108, 109, 110, 111, 112, 353, 354, 355, 356, 357, 358, 359, 360, 658, 113, 114, 115, 116, 117, 118, 119, 120, 361, 362, 363, 364, 365, 366, 367, 368, 659, 113, 114, 115, 116, 117, 118, 119, 120, 361, 362, 363, 364, 365, 366, 367, 368, 660, 113, 114, 115, 116, 117, 118, 119, 120, 361, 362, 363, 364, 365, 366, 367, 368, 661, 113, 114, 115, 116, 117, 118, 119, 120, 361, 362, 363, 364, 365, 366, 367, 368, 662, 113, 114, 115, 116, 117, 118, 119, 120, 361, 362, 363, 364, 365, 366, 367, 368, 663, 113, 114, 115, 116, 117, 118, 119, 120, 361, 362, 363, 364, 365, 366, 367, 368, 664, 113, 114, 115, 116, 117, 118, 119, 120, 361, 362, 363, 364, 365, 366, 367, 368, 665, 113, 114, 115, 116, 117, 118, 119, 120, 361, 362, 363, 364, 365, 366, 367, 368, 666, 121, 457, 667, 122, 458, 668, 123, 459, 669, 124, 460, 670, 125, 461, 671, 126, 462, 672, 127, 463, 673, 128, 464, 674, 129, 409, 675, 130, 410, 676, 131, 411, 677, 132, 412, 678, 133, 413, 679, 134, 414, 680, 135, 415, 681, 136, 416, 682, 137, 425, 683, 138, 426, 684, 139, 427, 685, 140, 428, 686, 141, 429, 687, 142, 430, 688, 143, 431, 689, 144, 432, 690, 145, 441, 691, 146, 442, 692, 147, 443, 693, 148, 444, 694, 149, 445, 695, 150, 446, 696, 151, 447, 697, 152, 448, 698, 153, 449, 699, 154, 450, 700, 155, 451, 701, 156, 452, 702, 157, 453, 703, 158, 454, 704, 159, 455, 705, 160, 456, 706, 161, 433, 515, 707, 805, 162, 434, 516, 708, 163, 435, 517, 709, 164, 436, 518, 710, 165, 437, 519, 711, 166, 438, 520, 712, 167, 439, 465, 521, 713, 806, 168, 440, 466, 522, 714, 169, 177, 475, 715, 841, 170, 178, 476, 716, 171, 179, 477, 717, 172, 180, 478, 718, 173, 181, 479, 719, 174, 182, 480, 720, 175, 183, 481, 721, 842, 176, 184, 482, 722, 169, 177, 723, 170, 178, 724, 171, 179, 725, 172, 180, 726, 173, 181, 727, 174, 182, 728, 175, 183, 729, 176, 184, 730, 185, 507, 731, 186, 508, 732, 187, 509, 733, 188, 510, 734, 189, 511, 735, 190, 512, 736, 191, 513, 737, 192, 514, 738, 193, 499, 739, 194, 500, 740, 195, 501, 741, 196, 502, 742, 197, 503, 743, 198, 504, 744, 199, 505, 745, 200, 506, 746, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 787, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 788, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 789, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 790, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 791, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 792, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 793, 49, 50, 51, 52, 53, 54, 55, 56, 201, 202, 203, 204, 205, 206, 207, 208, 794, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 1, 2, 3, 4, 5, 6, 7, 8, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 9, 10, 11, 12, 13, 14, 15, 16, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 9, 10, 11, 12, 13, 14, 15, 16, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 9, 10, 11, 12, 13, 14, 15, 16, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 9, 10, 11, 12, 13, 14, 15, 16, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 9, 10, 11, 12, 13, 14, 15, 16, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 9, 10, 11, 12, 13, 14, 15, 16, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 9, 10, 11, 12, 13, 14, 15, 16, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 9, 10, 11, 12, 13, 14, 15, 16, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 281, 282, 283, 284, 285, 286, 287, 288, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 281, 282, 283, 284, 285, 286, 287, 288, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 281, 282, 283, 284, 285, 286, 287, 288, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 281, 282, 283, 284, 285, 286, 287, 288, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 281, 282, 283, 284, 285, 286, 287, 288, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 281, 282, 283, 284, 285, 286, 287, 288, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 281, 282, 283, 284, 285, 286, 287, 288, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 281, 282, 283, 284, 285, 286, 287, 288, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 25, 26, 27, 28, 29, 30, 31, 32, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 25, 26, 27, 28, 29, 30, 31, 32, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 25, 26, 27, 28, 29, 30, 31, 32, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 25, 26, 27, 28, 29, 30, 31, 32, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 25, 26, 27, 28, 29, 30, 31, 32, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 25, 26, 27, 28, 29, 30, 31, 32, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 25, 26, 27, 28, 29, 30, 31, 32, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 25, 26, 27, 28, 29, 30, 31, 32, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 33, 34, 35, 36, 37, 38, 39, 40, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 33, 34, 35, 36, 37, 38, 39, 40, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 33, 34, 35, 36, 37, 38, 39, 40, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 33, 34, 35, 36, 37, 38, 39, 40, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 33, 34, 35, 36, 37, 38, 39, 40, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 33, 34, 35, 36, 37, 38, 39, 40, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 33, 34, 35, 36, 37, 38, 39, 40, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 33, 34, 35, 36, 37, 38, 39, 40, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 41, 42, 43, 44, 45, 46, 47, 48, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 759, 41, 42, 43, 44, 45, 46, 47, 48, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 760, 41, 42, 43, 44, 45, 46, 47, 48, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 761, 41, 42, 43, 44, 45, 46, 47, 48, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 762, 41, 42, 43, 44, 45, 46, 47, 48, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 763, 41, 42, 43, 44, 45, 46, 47, 48, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 764, 41, 42, 43, 44, 45, 46, 47, 48, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 765, 41, 42, 43, 44, 45, 46, 47, 48, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 766, 225, 226, 227, 228, 229, 230, 231, 232, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 17, 18, 19, 20, 21, 22, 23, 24, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 771, 17, 18, 19, 20, 21, 22, 23, 24, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 772, 17, 18, 19, 20, 21, 22, 23, 24, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 773, 17, 18, 19, 20, 21, 22, 23, 24, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 774, 17, 18, 19, 20, 21, 22, 23, 24, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 775, 17, 18, 19, 20, 21, 22, 23, 24, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 776, 17, 18, 19, 20, 21, 22, 23, 24, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 777, 17, 18, 19, 20, 21, 22, 23, 24, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 778, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 57, 58, 59, 60, 61, 62, 63, 64, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 369, 370, 371, 372, 373, 374, 375, 376, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 369, 370, 371, 372, 373, 374, 375, 376, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 369, 370, 371, 372, 373, 374, 375, 376, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 369, 370, 371, 372, 373, 374, 375, 376, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 369, 370, 371, 372, 373, 374, 375, 376, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 369, 370, 371, 372, 373, 374, 375, 376, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 369, 370, 371, 372, 373, 374, 375, 376, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 369, 370, 371, 372, 373, 374, 375, 376, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 385, 386, 387, 388, 389, 390, 391, 392, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 385, 386, 387, 388, 389, 390, 391, 392, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 385, 386, 387, 388, 389, 390, 391, 392, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 385, 386, 387, 388, 389, 390, 391, 392, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 385, 386, 387, 388, 389, 390, 391, 392, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 385, 386, 387, 388, 389, 390, 391, 392, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 385, 386, 387, 388, 389, 390, 391, 392, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 385, 386, 387, 388, 389, 390, 391, 392, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 105, 106, 107, 108, 109, 110, 111, 112, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 105, 106, 107, 108, 109, 110, 111, 112, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 105, 106, 107, 108, 109, 110, 111, 112, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 105, 106, 107, 108, 109, 110, 111, 112, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 105, 106, 107, 108, 109, 110, 111, 112, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 105, 106, 107, 108, 109, 110, 111, 112, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 105, 106, 107, 108, 109, 110, 111, 112, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 105, 106, 107, 108, 109, 110, 111, 112, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 113, 114, 115, 116, 117, 118, 119, 120, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 821, 113, 114, 115, 116, 117, 118, 119, 120, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 822, 113, 114, 115, 116, 117, 118, 119, 120, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 823, 113, 114, 115, 116, 117, 118, 119, 120, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 824, 113, 114, 115, 116, 117, 118, 119, 120, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 825, 113, 114, 115, 116, 117, 118, 119, 120, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 826, 113, 114, 115, 116, 117, 118, 119, 120, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 827, 113, 114, 115, 116, 117, 118, 119, 120, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 828, 81, 82, 83, 84, 85, 86, 87, 88, 313, 314, 315, 316, 317, 318, 319, 320, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 81, 82, 83, 84, 85, 86, 87, 88, 313, 314, 315, 316, 317, 318, 319, 320, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 81, 82, 83, 84, 85, 86, 87, 88, 313, 314, 315, 316, 317, 318, 319, 320, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 81, 82, 83, 84, 85, 86, 87, 88, 313, 314, 315, 316, 317, 318, 319, 320, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 81, 82, 83, 84, 85, 86, 87, 88, 313, 314, 315, 316, 317, 318, 319, 320, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 81, 82, 83, 84, 85, 86, 87, 88, 313, 314, 315, 316, 317, 318, 319, 320, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 81, 82, 83, 84, 85, 86, 87, 88, 313, 314, 315, 316, 317, 318, 319, 320, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 81, 82, 83, 84, 85, 86, 87, 88, 313, 314, 315, 316, 317, 318, 319, 320, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 833, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 834, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 835, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 836, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 837, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 838, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 839, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 840, 97, 98, 99, 100, 101, 102, 103, 104, 329, 330, 331, 332, 333, 334, 335, 336, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 97, 98, 99, 100, 101, 102, 103, 104, 329, 330, 331, 332, 333, 334, 335, 336, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 97, 98, 99, 100, 101, 102, 103, 104, 329, 330, 331, 332, 333, 334, 335, 336, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 97, 98, 99, 100, 101, 102, 103, 104, 329, 330, 331, 332, 333, 334, 335, 336, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 97, 98, 99, 100, 101, 102, 103, 104, 329, 330, 331, 332, 333, 334, 335, 336, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 97, 98, 99, 100, 101, 102, 103, 104, 329, 330, 331, 332, 333, 334, 335, 336, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 97, 98, 99, 100, 101, 102, 103, 104, 329, 330, 331, 332, 333, 334, 335, 336, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 97, 98, 99, 100, 101, 102, 103, 104, 329, 330, 331, 332, 333, 334, 335, 336, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 89, 90, 91, 92, 93, 94, 95, 96, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 845, 89, 90, 91, 92, 93, 94, 95, 96, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 846, 89, 90, 91, 92, 93, 94, 95, 96, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 847, 89, 90, 91, 92, 93, 94, 95, 96, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 848, 89, 90, 91, 92, 93, 94, 95, 96, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 849, 89, 90, 91, 92, 93, 94, 95, 96, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 850, 89, 90, 91, 92, 93, 94, 95, 96, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 851, 89, 90, 91, 92, 93, 94, 95, 96, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 852, 401, 409, 755, 402, 410, 403, 411, 404, 412, 405, 413, 406, 414, 407, 415, 756, 408, 416, 129, 401, 409, 417, 130, 402, 410, 418, 131, 403, 411, 419, 132, 404, 412, 420, 133, 405, 413, 421, 134, 406, 414, 422, 135, 407, 415, 423, 136, 408, 416, 424, 409, 417, 425, 457, 410, 418, 426, 458, 411, 419, 427, 459, 412, 420, 428, 460, 413, 421, 429, 461, 414, 422, 430, 462, 415, 423, 431, 463, 416, 424, 432, 464, 137, 417, 425, 433, 138, 418, 426, 434, 139, 419, 427, 435, 140, 420, 428, 436, 141, 421, 429, 437, 142, 422, 430, 438, 143, 423, 431, 439, 144, 424, 432, 440, 161, 425, 433, 441, 162, 426, 434, 442, 163, 427, 435, 443, 164, 428, 436, 444, 165, 429, 437, 445, 166, 430, 438, 446, 167, 431, 439, 447, 168, 432, 440, 448, 145, 433, 441, 449, 146, 434, 442, 450, 147, 435, 443, 451, 148, 436, 444, 452, 149, 437, 445, 453, 150, 438, 446, 454, 151, 439, 447, 455, 152, 440, 448, 456, 153, 441, 449, 767, 154, 442, 450, 155, 443, 451, 156, 444, 452, 157, 445, 453, 158, 446, 454, 159, 447, 455, 768, 160, 448, 456, 121, 417, 457, 779, 122, 418, 458, 123, 419, 459, 124, 420, 460, 125, 421, 461, 126, 422, 462, 127, 423, 463, 780, 128, 424, 464, 167, 465, 795, 168, 466, 467, 475, 817, 468, 476, 469, 477, 470, 478, 471, 479, 472, 480, 473, 481, 818, 474, 482, 169, 467, 475, 483, 170, 468, 476, 484, 171, 469, 477, 485, 172, 470, 478, 486, 173, 471, 479, 487, 174, 472, 480, 488, 175, 473, 481, 489, 176, 474, 482, 490, 475, 483, 491, 507, 476, 484, 492, 508, 477, 485, 493, 509, 478, 486, 494, 510, 479, 487, 495, 511, 480, 488, 496, 512, 481, 489, 497, 513, 482, 490, 498, 514, 483, 491, 499, 515, 484, 492, 500, 516, 485, 493, 501, 517, 486, 494, 502, 518, 487, 495, 503, 519, 488, 496, 504, 520, 489, 497, 505, 521, 490, 498, 506, 522, 193, 491, 499, 829, 194, 492, 500, 195, 493, 501, 196, 494, 502, 197, 495, 503, 198, 496, 504, 199, 497, 505, 830, 200, 498, 506, 185, 483, 507, 853, 186, 484, 508, 187, 485, 509, 188, 486, 510, 189, 487, 511, 190, 488, 512, 191, 489, 513, 854, 192, 490, 514, 161, 491, 515, 162, 492, 516, 163, 493, 517, 164, 494, 518, 165, 495, 519, 166, 496, 520, 167, 497, 521, 168, 498, 522, 401, 409, 523, 402, 410, 524, 403, 411, 525, 404, 412, 526, 405, 413, 527, 406, 414, 528, 523, 525, 527, 529, 524, 526, 528, 530, 161, 531, 162, 532, 163, 533, 164, 534, 165, 535, 166, 536, 531, 533, 535, 537, 532, 534, 536, 538, 467, 475, 539, 468, 476, 540, 469, 477, 541, 470, 478, 542, 471, 479, 543, 472, 480, 544, 539, 541, 543, 545, 540, 542, 544, 546, 1, 2, 7, 8, 547, 548, 3, 4, 7, 8, 549, 550, 5, 6, 7, 8, 551, 552, 1, 2, 7, 8, 547, 548, 3, 4, 7, 8, 549, 550, 5, 6, 7, 8, 551, 552, 547, 549, 551, 553, 548, 550, 552, 554, 9, 10, 15, 16, 555, 556, 11, 12, 15, 16, 557, 558, 13, 14, 15, 16, 559, 560, 9, 10, 15, 16, 555, 556, 11, 12, 15, 16, 557, 558, 13, 14, 15, 16, 559, 560, 555, 557, 559, 561, 556, 558, 560, 562, 17, 18, 23, 24, 563, 564, 19, 20, 23, 24, 565, 566, 21, 22, 23, 24, 567, 568, 17, 18, 23, 24, 563, 564, 19, 20, 23, 24, 565, 566, 21, 22, 23, 24, 567, 568, 563, 565, 567, 569, 564, 566, 568, 570, 25, 26, 31, 32, 571, 572, 27, 28, 31, 32, 573, 574, 29, 30, 31, 32, 575, 576, 25, 26, 31, 32, 571, 572, 27, 28, 31, 32, 573, 574, 29, 30, 31, 32, 575, 576, 571, 573, 575, 577, 572, 574, 576, 578, 33, 34, 39, 40, 579, 580, 35, 36, 39, 40, 581, 582, 37, 38, 39, 40, 583, 584, 33, 34, 39, 40, 579, 580, 35, 36, 39, 40, 581, 582, 37, 38, 39, 40, 583, 584, 579, 581, 583, 585, 580, 582, 584, 586, 41, 42, 47, 48, 587, 588, 43, 44, 47, 48, 589, 590, 45, 46, 47, 48, 591, 592, 41, 42, 47, 48, 587, 588, 43, 44, 47, 48, 589, 590, 45, 46, 47, 48, 591, 592, 587, 589, 591, 593, 588, 590, 592, 594, 49, 50, 55, 56, 595, 596, 51, 52, 55, 56, 597, 598, 53, 54, 55, 56, 599, 600, 49, 50, 55, 56, 595, 596, 51, 52, 55, 56, 597, 598, 53, 54, 55, 56, 599, 600, 595, 597, 599, 601, 596, 598, 600, 602, 57, 58, 63, 64, 603, 604, 59, 60, 63, 64, 605, 606, 61, 62, 63, 64, 607, 608, 57, 58, 63, 64, 603, 604, 59, 60, 63, 64, 605, 606, 61, 62, 63, 64, 607, 608, 603, 605, 607, 609, 604, 606, 608, 610, 65, 66, 71, 72, 611, 612, 67, 68, 71, 72, 613, 614, 69, 70, 71, 72, 615, 616, 65, 66, 71, 72, 611, 612, 67, 68, 71, 72, 613, 614, 69, 70, 71, 72, 615, 616, 611, 613, 615, 617, 612, 614, 616, 618, 73, 74, 79, 80, 619, 620, 75, 76, 79, 80, 621, 622, 77, 78, 79, 80, 623, 624, 73, 74, 79, 80, 619, 620, 75, 76, 79, 80, 621, 622, 77, 78, 79, 80, 623, 624, 619, 621, 623, 625, 620, 622, 624, 626, 81, 82, 87, 88, 627, 628, 83, 84, 87, 88, 629, 630, 85, 86, 87, 88, 631, 632, 81, 82, 87, 88, 627, 628, 83, 84, 87, 88, 629, 630, 85, 86, 87, 88, 631, 632, 627, 629, 631, 633, 628, 630, 632, 634, 89, 90, 95, 96, 635, 636, 91, 92, 95, 96, 637, 638, 93, 94, 95, 96, 639, 640, 89, 90, 95, 96, 635, 636, 91, 92, 95, 96, 637, 638, 93, 94, 95, 96, 639, 640, 635, 637, 639, 641, 636, 638, 640, 642, 97, 98, 103, 104, 643, 644, 99, 100, 103, 104, 645, 646, 101, 102, 103, 104, 647, 648, 97, 98, 103, 104, 643, 644, 99, 100, 103, 104, 645, 646, 101, 102, 103, 104, 647, 648, 643, 645, 647, 649, 644, 646, 648, 650, 105, 106, 111, 112, 651, 652, 107, 108, 111, 112, 653, 654, 109, 110, 111, 112, 655, 656, 105, 106, 111, 112, 651, 652, 107, 108, 111, 112, 653, 654, 109, 110, 111, 112, 655, 656, 651, 653, 655, 657, 652, 654, 656, 658, 113, 114, 119, 120, 659, 660, 115, 116, 119, 120, 661, 662, 117, 118, 119, 120, 663, 664, 113, 114, 119, 120, 659, 660, 115, 116, 119, 120, 661, 662, 117, 118, 119, 120, 663, 664, 659, 661, 663, 665, 660, 662, 664, 666, 121, 122, 127, 128, 667, 668, 123, 124, 127, 128, 669, 670, 125, 126, 127, 128, 671, 672, 121, 122, 127, 128, 667, 668, 123, 124, 127, 128, 669, 670, 125, 126, 127, 128, 671, 672, 667, 669, 671, 673, 668, 670, 672, 674, 129, 130, 135, 136, 675, 676, 131, 132, 135, 136, 677, 678, 133, 134, 135, 136, 679, 680, 129, 130, 135, 136, 675, 676, 131, 132, 135, 136, 677, 678, 133, 134, 135, 136, 679, 680, 675, 677, 679, 681, 676, 678, 680, 682, 137, 138, 143, 144, 683, 684, 139, 140, 143, 144, 685, 686, 141, 142, 143, 144, 687, 688, 137, 138, 143, 144, 683, 684, 139, 140, 143, 144, 685, 686, 141, 142, 143, 144, 687, 688, 683, 685, 687, 689, 684, 686, 688, 690, 145, 146, 151, 152, 691, 692, 147, 148, 151, 152, 693, 694, 149, 150, 151, 152, 695, 696, 145, 146, 151, 152, 691, 692, 147, 148, 151, 152, 693, 694, 149, 150, 151, 152, 695, 696, 691, 693, 695, 697, 692, 694, 696, 698, 153, 154, 159, 160, 699, 700, 155, 156, 159, 160, 701, 702, 157, 158, 159, 160, 703, 704, 153, 154, 159, 160, 699, 700, 155, 156, 159, 160, 701, 702, 157, 158, 159, 160, 703, 704, 699, 701, 703, 705, 700, 702, 704, 706, 161, 162, 167, 168, 707, 708, 163, 164, 167, 168, 709, 710, 165, 166, 167, 168, 711, 712, 161, 162, 167, 168, 707, 708, 163, 164, 167, 168, 709, 710, 165, 166, 167, 168, 711, 712, 707, 709, 711, 713, 708, 710, 712, 714, 169, 170, 175, 176, 715, 716, 171, 172, 175, 176, 717, 718, 173, 174, 175, 176, 719, 720, 169, 170, 175, 176, 715, 716, 171, 172, 175, 176, 717, 718, 173, 174, 175, 176, 719, 720, 715, 717, 719, 721, 716, 718, 720, 722, 177, 178, 183, 184, 723, 724, 179, 180, 183, 184, 725, 726, 181, 182, 183, 184, 727, 728, 177, 178, 183, 184, 723, 724, 179, 180, 183, 184, 725, 726, 181, 182, 183, 184, 727, 728, 723, 725, 727, 729, 724, 726, 728, 730, 185, 186, 191, 192, 731, 732, 187, 188, 191, 192, 733, 734, 189, 190, 191, 192, 735, 736, 185, 186, 191, 192, 731, 732, 187, 188, 191, 192, 733, 734, 189, 190, 191, 192, 735, 736, 731, 733, 735, 737, 732, 734, 736, 738, 193, 194, 199, 200, 739, 740, 195, 196, 199, 200, 741, 742, 197, 198, 199, 200, 743, 744, 193, 194, 199, 200, 739, 740, 195, 196, 199, 200, 741, 742, 197, 198, 199, 200, 743, 744, 739, 741, 743, 745, 740, 742, 744, 746, 1, 2, 7, 8, 747, 748, 1, 2, 7, 8, 747, 748, 3, 4, 7, 8, 749, 750, 3, 4, 7, 8, 749, 750, 5, 6, 7, 8, 751, 752, 5, 6, 7, 8, 751, 752, 747, 749, 751, 753, 748, 750, 752, 754, 401, 407, 755, 757, 401, 407, 756, 757, 757, 758, 747, 748, 749, 750, 751, 752, 753, 754, 758, 273, 274, 279, 280, 759, 760, 273, 274, 279, 280, 759, 760, 275, 276, 279, 280, 761, 762, 275, 276, 279, 280, 761, 762, 277, 278, 279, 280, 763, 764, 277, 278, 279, 280, 763, 764, 759, 761, 763, 765, 760, 762, 764, 766, 449, 455, 767, 769, 449, 455, 768, 769, 769, 770, 759, 760, 761, 762, 763, 764, 765, 766, 770, 297, 298, 303, 304, 771, 772, 297, 298, 303, 304, 771, 772, 299, 300, 303, 304, 773, 774, 299, 300, 303, 304, 773, 774, 301, 302, 303, 304, 775, 776, 301, 302, 303, 304, 775, 776, 771, 773, 775, 777, 772, 774, 776, 778, 457, 463, 779, 781, 457, 463, 780, 781, 781, 782, 771, 772, 773, 774, 775, 776, 777, 778, 782, 783, 796, 784, 796, 785, 796, 207, 208, 786, 793, 794, 201, 202, 207, 208, 783, 787, 788, 793, 794, 201, 202, 207, 208, 787, 788, 203, 204, 207, 208, 784, 789, 790, 203, 204, 207, 208, 789, 790, 205, 206, 207, 208, 785, 791, 792, 795, 205, 206, 207, 208, 791, 792, 787, 789, 791, 793, 788, 790, 792, 794, 465, 795, 795, 796, 49, 50, 55, 56, 797, 798, 49, 50, 55, 56, 797, 798, 51, 52, 55, 56, 799, 800, 51, 52, 55, 56, 799, 800, 53, 54, 55, 56, 801, 802, 53, 54, 55, 56, 801, 802, 797, 799, 801, 803, 798, 800, 802, 804, 161, 167, 805, 807, 161, 167, 806, 807, 807, 808, 797, 798, 799, 800, 801, 802, 803, 804, 808, 57, 58, 63, 64, 809, 810, 57, 58, 63, 64, 809, 810, 59, 60, 63, 64, 811, 812, 59, 60, 63, 64, 811, 812, 61, 62, 63, 64, 813, 814, 61, 62, 63, 64, 813, 814, 809, 811, 813, 815, 810, 812, 814, 816, 467, 473, 817, 819, 467, 473, 818, 819, 819, 820, 809, 810, 811, 812, 813, 814, 815, 816, 820, 361, 362, 367, 368, 821, 822, 361, 362, 367, 368, 821, 822, 363, 364, 367, 368, 823, 824, 363, 364, 367, 368, 823, 824, 365, 366, 367, 368, 825, 826, 365, 366, 367, 368, 825, 826, 821, 823, 825, 827, 822, 824, 826, 828, 499, 505, 829, 831, 499, 505, 830, 831, 831, 832, 821, 822, 823, 824, 825, 826, 827, 828, 832, 377, 378, 383, 384, 833, 834, 377, 378, 383, 384, 833, 834, 379, 380, 383, 384, 835, 836, 379, 380, 383, 384, 835, 836, 381, 382, 383, 384, 837, 838, 381, 382, 383, 384, 837, 838, 833, 835, 837, 839, 834, 836, 838, 840, 169, 175, 841, 843, 169, 175, 842, 843, 843, 844, 833, 834, 835, 836, 837, 838, 839, 840, 844, 393, 394, 399, 400, 845, 846, 393, 394, 399, 400, 845, 846, 395, 396, 399, 400, 847, 848, 395, 396, 399, 400, 847, 848, 397, 398, 399, 400, 849, 850, 397, 398, 399, 400, 849, 850, 845, 847, 849, 851, 846, 848, 850, 852, 507, 513, 853, 855, 507, 513, 854, 855, 855, 856, 845, 846, 847, 848, 849, 850, 851, 852, 856]
    sp_jac_trap_ja = [0, 1, 19, 37, 55, 73, 91, 109, 127, 145, 162, 179, 196, 213, 230, 247, 264, 281, 298, 315, 332, 349, 366, 383, 400, 417, 434, 451, 468, 485, 502, 519, 536, 553, 570, 587, 604, 621, 638, 655, 672, 689, 706, 723, 740, 757, 774, 791, 808, 825, 843, 861, 879, 897, 915, 933, 951, 969, 987, 1005, 1023, 1041, 1059, 1077, 1095, 1113, 1130, 1147, 1164, 1181, 1198, 1215, 1232, 1249, 1266, 1283, 1300, 1317, 1334, 1351, 1368, 1385, 1402, 1419, 1436, 1453, 1470, 1487, 1504, 1521, 1538, 1555, 1572, 1589, 1606, 1623, 1640, 1657, 1674, 1691, 1708, 1725, 1742, 1759, 1776, 1793, 1810, 1827, 1844, 1861, 1878, 1895, 1912, 1929, 1946, 1963, 1980, 1997, 2014, 2031, 2048, 2065, 2068, 2071, 2074, 2077, 2080, 2083, 2086, 2089, 2092, 2095, 2098, 2101, 2104, 2107, 2110, 2113, 2116, 2119, 2122, 2125, 2128, 2131, 2134, 2137, 2140, 2143, 2146, 2149, 2152, 2155, 2158, 2161, 2164, 2167, 2170, 2173, 2176, 2179, 2182, 2185, 2190, 2194, 2198, 2202, 2206, 2210, 2216, 2221, 2226, 2230, 2234, 2238, 2242, 2246, 2251, 2255, 2258, 2261, 2264, 2267, 2270, 2273, 2276, 2279, 2282, 2285, 2288, 2291, 2294, 2297, 2300, 2303, 2306, 2309, 2312, 2315, 2318, 2321, 2324, 2327, 2344, 2361, 2378, 2395, 2412, 2429, 2446, 2463, 2487, 2511, 2535, 2559, 2583, 2607, 2631, 2655, 2687, 2719, 2751, 2783, 2815, 2847, 2879, 2911, 2943, 2975, 3007, 3039, 3071, 3103, 3135, 3167, 3191, 3215, 3239, 3263, 3287, 3311, 3335, 3359, 3391, 3423, 3455, 3487, 3519, 3551, 3583, 3615, 3639, 3663, 3687, 3711, 3735, 3759, 3783, 3807, 3831, 3855, 3879, 3903, 3927, 3951, 3975, 3999, 4031, 4063, 4095, 4127, 4159, 4191, 4223, 4255, 4280, 4305, 4330, 4355, 4380, 4405, 4430, 4455, 4479, 4503, 4527, 4551, 4575, 4599, 4623, 4647, 4671, 4695, 4719, 4743, 4767, 4791, 4815, 4839, 4864, 4889, 4914, 4939, 4964, 4989, 5014, 5039, 5063, 5087, 5111, 5135, 5159, 5183, 5207, 5231, 5263, 5295, 5327, 5359, 5391, 5423, 5455, 5487, 5511, 5535, 5559, 5583, 5607, 5631, 5655, 5679, 5711, 5743, 5775, 5807, 5839, 5871, 5903, 5935, 5959, 5983, 6007, 6031, 6055, 6079, 6103, 6127, 6151, 6175, 6199, 6223, 6247, 6271, 6295, 6319, 6351, 6383, 6415, 6447, 6479, 6511, 6543, 6575, 6600, 6625, 6650, 6675, 6700, 6725, 6750, 6775, 6807, 6839, 6871, 6903, 6935, 6967, 6999, 7031, 7064, 7097, 7130, 7163, 7196, 7229, 7262, 7295, 7327, 7359, 7391, 7423, 7455, 7487, 7519, 7551, 7576, 7601, 7626, 7651, 7676, 7701, 7726, 7751, 7754, 7756, 7758, 7760, 7762, 7764, 7767, 7769, 7773, 7777, 7781, 7785, 7789, 7793, 7797, 7801, 7805, 7809, 7813, 7817, 7821, 7825, 7829, 7833, 7837, 7841, 7845, 7849, 7853, 7857, 7861, 7865, 7869, 7873, 7877, 7881, 7885, 7889, 7893, 7897, 7901, 7905, 7909, 7913, 7917, 7921, 7925, 7929, 7933, 7936, 7939, 7942, 7945, 7948, 7952, 7955, 7959, 7962, 7965, 7968, 7971, 7974, 7978, 7981, 7984, 7986, 7989, 7991, 7993, 7995, 7997, 7999, 8002, 8004, 8008, 8012, 8016, 8020, 8024, 8028, 8032, 8036, 8040, 8044, 8048, 8052, 8056, 8060, 8064, 8068, 8072, 8076, 8080, 8084, 8088, 8092, 8096, 8100, 8104, 8107, 8110, 8113, 8116, 8119, 8123, 8126, 8130, 8133, 8136, 8139, 8142, 8145, 8149, 8152, 8155, 8158, 8161, 8164, 8167, 8170, 8173, 8176, 8179, 8182, 8185, 8188, 8191, 8194, 8198, 8202, 8204, 8206, 8208, 8210, 8212, 8214, 8218, 8222, 8225, 8228, 8231, 8234, 8237, 8240, 8244, 8248, 8254, 8260, 8266, 8272, 8278, 8284, 8288, 8292, 8298, 8304, 8310, 8316, 8322, 8328, 8332, 8336, 8342, 8348, 8354, 8360, 8366, 8372, 8376, 8380, 8386, 8392, 8398, 8404, 8410, 8416, 8420, 8424, 8430, 8436, 8442, 8448, 8454, 8460, 8464, 8468, 8474, 8480, 8486, 8492, 8498, 8504, 8508, 8512, 8518, 8524, 8530, 8536, 8542, 8548, 8552, 8556, 8562, 8568, 8574, 8580, 8586, 8592, 8596, 8600, 8606, 8612, 8618, 8624, 8630, 8636, 8640, 8644, 8650, 8656, 8662, 8668, 8674, 8680, 8684, 8688, 8694, 8700, 8706, 8712, 8718, 8724, 8728, 8732, 8738, 8744, 8750, 8756, 8762, 8768, 8772, 8776, 8782, 8788, 8794, 8800, 8806, 8812, 8816, 8820, 8826, 8832, 8838, 8844, 8850, 8856, 8860, 8864, 8870, 8876, 8882, 8888, 8894, 8900, 8904, 8908, 8914, 8920, 8926, 8932, 8938, 8944, 8948, 8952, 8958, 8964, 8970, 8976, 8982, 8988, 8992, 8996, 9002, 9008, 9014, 9020, 9026, 9032, 9036, 9040, 9046, 9052, 9058, 9064, 9070, 9076, 9080, 9084, 9090, 9096, 9102, 9108, 9114, 9120, 9124, 9128, 9134, 9140, 9146, 9152, 9158, 9164, 9168, 9172, 9178, 9184, 9190, 9196, 9202, 9208, 9212, 9216, 9222, 9228, 9234, 9240, 9246, 9252, 9256, 9260, 9266, 9272, 9278, 9284, 9290, 9296, 9300, 9304, 9310, 9316, 9322, 9328, 9334, 9340, 9344, 9348, 9354, 9360, 9366, 9372, 9378, 9384, 9388, 9392, 9396, 9400, 9402, 9411, 9417, 9423, 9429, 9435, 9441, 9447, 9451, 9455, 9459, 9463, 9465, 9474, 9480, 9486, 9492, 9498, 9504, 9510, 9514, 9518, 9522, 9526, 9528, 9537, 9539, 9541, 9543, 9548, 9557, 9563, 9570, 9576, 9584, 9590, 9594, 9598, 9600, 9602, 9608, 9614, 9620, 9626, 9632, 9638, 9642, 9646, 9650, 9654, 9656, 9665, 9671, 9677, 9683, 9689, 9695, 9701, 9705, 9709, 9713, 9717, 9719, 9728, 9734, 9740, 9746, 9752, 9758, 9764, 9768, 9772, 9776, 9780, 9782, 9791, 9797, 9803, 9809, 9815, 9821, 9827, 9831, 9835, 9839, 9843, 9845, 9854, 9860, 9866, 9872, 9878, 9884, 9890, 9894, 9898, 9902, 9906, 9908, 9917]
    sp_jac_trap_nia = 857
    sp_jac_trap_nja = 857
    return sp_jac_trap_ia, sp_jac_trap_ja, sp_jac_trap_nia, sp_jac_trap_nja 
