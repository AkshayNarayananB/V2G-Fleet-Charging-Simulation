from simulator.battery import *

def test_multi_stage_battery():
    bat = MultiStageBattery(100)
    assert bat.actual_capacity == bat.initial_capacity
    assert bat.soc == 1
    bat.discharge(100, 3600, 25)
    assert bat.soc == 0

