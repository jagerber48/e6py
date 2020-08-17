

def tfODT_volt_to_pow(volt):
    # Convert Cicero Voltage (V) to power (W)
    # Calibrated Feb 12, 2019, code written Apr 5, 2019
    return 1.7*volt - 0.25
