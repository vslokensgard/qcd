import pionkaon as pk
pairs = [[0, ["2pt_0"]], [1, ["2pt_1"]]]
gs = pk.listGroundStates(pairs, "./2pt_data/", 4, mode=0)
print(gs)
pk.plotDispersion([0,1], gs)
