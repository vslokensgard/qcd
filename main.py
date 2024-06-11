pairs = [[0, ["2pt_0"]], [1, ["2pt_1"]]]
gs = listGroundStates(pairs, "./2pt_data/", 4, mode=0)
print(gs)
plotDispersion([0,1], gs)