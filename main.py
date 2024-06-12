import pionkaon as pk

query = "energy"
p = [ (0, [query, "zero"]), (1, [query, "one"]), (2, [query, "two"]), (3, [query, "three"]), (4, [query, "four"]), (5, [query, "five"]) ]
path = "./"
col_index = 4
ground_states = pk.listGroundStates(p, path, col_index, mode=0)
np.savetxt("default_settings", ground_states)
