import pionkaon as pk

p = [ (0, ["zero"]), (1, ["one"]), (2, ["two"]), (3, ["three"]), (4, ["four"]), (5, ["five"]) ]
path = "./"
col_index = 4
set_mins = [9, 10, 6, 6, 6, 6]
set_maxs = [14, 20, 9, 11, 11, 10]
read_from = "data-files/data-energy/"
save_to = "export/"
pk.exportBins(p, col_index, read_from, save_to, debug=True)
#ground_states = pk.listGroundStates(p, "data-energy/", col_index, mode=1)
#ground_states = pk.listGroundStates(p, "data-energy/", col_index, mode=0, set_mins=set_mins, set_maxs=set_maxs)
