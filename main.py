import pionkaon as pk

p = [ (0, ["zero"]), (1, ["one"]), (2, ["two"]), (3, ["three"]), (4, ["four"]), (5, ["five"]) ]
path = "./"
col_index = 4
ground_states = pk.listGroundStates(p, "data-energy/", col_index, mode=0)
