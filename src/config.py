ignore_labels = [
    'lv_sys_func_unchanged',
    'lv_sys_func_improved',
    'lv_sys_func_unknown',
]

replace_list = {
    'aortic_valve_native_stenosis_not_present': 'aortic_valve_native_regurgitation_not_present',
    'tricuspid_native_regurgitation_not_present': 'tricuspid_valve_native_regurgitation_not_present',
    'lv_sys_func_hyperdynamic': 'lv_sys_func_normal', # Hyper dynamisch kan cool zijn maar we hebben te weinig data
    'rv_sys_func_hyperdynamic': 'rv_sys_func_normal'
}
